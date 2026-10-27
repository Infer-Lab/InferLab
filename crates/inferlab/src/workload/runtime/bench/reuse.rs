//! Per-request reuse opportunity of a controlled-start case over a generated
//! population ([[RFC-0004:C-BENCH-CACHE-STATE]]).
//!
//! Every entry of a generated population shares a nested prefix of one
//! canonical stream, so an earlier writer covers `min(its shared tokens, this
//! request's)` of a later request's shared prefix. The writers are the
//! conditioning request and every measured request sent before this one;
//! warmup traffic precedes the reset and writes nothing that survives it.

use std::collections::BTreeMap;

use inferlab_protocol::BenchPromptCacheObservation;

use crate::workload::record::{
    PrefixCacheConditioningEvidence, PromptReuseEvidence, PromptReuseObservation, ReuseOpportunity,
};
use crate::workspace::BenchCacheStart;

struct Writer {
    group: u32,
    shared_tokens: u64,
    sent_unix_ns: u64,
    published_unix_ns: u64,
}

/// The reuse opportunity of every measured request, or `None` when the case
/// has no defined writers (an uncontrolled start) or a request lacks the
/// generated-population facts it is derived from.
pub(super) fn prompt_reuse(
    start: BenchCacheStart,
    observations: &[BenchPromptCacheObservation],
    conditioning: Vec<&PrefixCacheConditioningEvidence>,
    publication_bound_ms: u64,
) -> Option<PromptReuseEvidence> {
    if start == BenchCacheStart::Uncontrolled || observations.is_empty() {
        return None;
    }
    let bound_ns = publication_bound_ms.saturating_mul(1_000_000);
    let mut requests = observations
        .iter()
        .map(|observation| {
            Some((
                observation.request_id,
                observation.prefix_group.unwrap_or(0),
                observation.shared_prefix_tokens?,
                observation.sent_unix_ns?,
                observation.first_token_unix_ns?,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    requests.sort_by_key(|&(request_id, _, _, sent, _)| (sent, request_id));
    let mut writers = Vec::with_capacity(requests.len() + conditioning.len());
    for conditioning in conditioning
        .into_iter()
        .filter(|conditioning| conditioning.succeeded)
    {
        if let (Some(completed), Some(shared)) = (
            conditioning.completed_unix_ns,
            conditioning.maximum_shared_prefix_tokens,
        ) {
            writers.push(Writer {
                group: conditioning.group,
                shared_tokens: u64::from(shared),
                sent_unix_ns: completed,
                published_unix_ns: completed.saturating_add(bound_ns),
            });
        }
    }
    let mut reuse = Vec::with_capacity(requests.len());
    for (request_id, group, shared_tokens, sent, first_token) in requests {
        // Only a writer of the same prefix group shares a prefix with it.
        let earlier = writers
            .iter()
            .filter(|writer| writer.group == group && writer.sent_unix_ns < sent)
            .map(|writer| {
                (
                    writer.shared_tokens.min(shared_tokens),
                    writer.published_unix_ns,
                )
            });
        let reusable_tokens = earlier
            .clone()
            .map(|(covered, _)| covered)
            .max()
            .unwrap_or(0);
        let opportunity = if reusable_tokens == 0 {
            ReuseOpportunity::First
        } else if earlier
            .clone()
            .any(|(covered, published)| covered >= reusable_tokens && published <= sent)
        {
            ReuseOpportunity::Available
        } else {
            ReuseOpportunity::InFlight
        };
        reuse.push(PromptReuseObservation {
            request_id,
            reusable_tokens,
            opportunity,
        });
        writers.push(Writer {
            group,
            shared_tokens,
            sent_unix_ns: sent,
            published_unix_ns: first_token.saturating_add(bound_ns),
        });
    }
    Some(PromptReuseEvidence {
        publication_bound_ms,
        requests: reuse,
    })
}

/// `available_miss_ratio` and `available_reuse_efficiency`, omitted when no
/// request is available ([[RFC-0004:C-BENCH-CACHE-STATE]]).
pub(super) fn reuse_metrics(
    evidence: &PromptReuseEvidence,
    observations: &[BenchPromptCacheObservation],
) -> BTreeMap<String, f64> {
    let cache_read = observations
        .iter()
        .map(|observation| (observation.request_id, observation.cache_read_tokens))
        .collect::<BTreeMap<_, _>>();
    let mut available = 0_u64;
    let mut missed = 0_u64;
    let mut reusable = 0_u64;
    let mut read = 0_u64;
    for request in &evidence.requests {
        if request.opportunity != ReuseOpportunity::Available {
            continue;
        }
        let tokens = cache_read.get(&request.request_id).copied().unwrap_or(0);
        available += 1;
        missed += u64::from(tokens == 0);
        reusable = reusable.saturating_add(request.reusable_tokens);
        read = read.saturating_add(tokens);
    }
    let mut metrics = BTreeMap::new();
    if available > 0 && reusable > 0 {
        metrics.insert(
            "available_miss_ratio".to_owned(),
            missed as f64 / available as f64,
        );
        metrics.insert(
            "available_reuse_efficiency".to_owned(),
            read as f64 / reusable as f64,
        );
    }
    metrics
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        request_id: u64,
        cache: u64,
        sent_ms: u64,
        ttft_ms: u64,
    ) -> BenchPromptCacheObservation {
        BenchPromptCacheObservation {
            request_id,
            prompt_tokens: 9216,
            cache_read_tokens: cache,
            uncached_prompt_tokens: 9216 - cache,
            cache_read_ratio: cache as f64 / 9216.0,
            shared_prefix_tokens: Some(8192),
            prefix_group: None,
            sent_unix_ns: Some(sent_ms * 1_000_000),
            first_token_unix_ns: Some((sent_ms + ttft_ms) * 1_000_000),
        }
    }

    /// A cold wave: the first request writes, the next arrives before it
    /// published, the third after — and misses, which the miss ratio exposes.
    #[test]
    fn a_cold_wave_separates_first_in_flight_and_available_requests() {
        let observations = [
            observation(0, 0, 0, 500),
            observation(1, 0, 100, 500),
            observation(2, 0, 2_000, 500),
            observation(3, 8064, 2_100, 100),
        ];

        let evidence = prompt_reuse(BenchCacheStart::Cold, &observations, Vec::new(), 1_000)
            .unwrap_or_else(|| PromptReuseEvidence {
                publication_bound_ms: 0,
                requests: Vec::new(),
            });

        let states = evidence
            .requests
            .iter()
            .map(|request| (request.reusable_tokens, request.opportunity))
            .collect::<Vec<_>>();
        assert_eq!(
            states,
            [
                (0, ReuseOpportunity::First),
                (8192, ReuseOpportunity::InFlight),
                (8192, ReuseOpportunity::Available),
                (8192, ReuseOpportunity::Available),
            ]
        );
        let metrics = reuse_metrics(&evidence, &observations);
        assert_eq!(metrics.get("available_miss_ratio"), Some(&0.5));
        assert_eq!(
            metrics.get("available_reuse_efficiency"),
            Some(&(8064.0 / 16384.0))
        );
    }

    /// Conditioning is the earliest writer of a primed start, so its first
    /// measured request is available once the bound has passed.
    #[test]
    fn a_primed_start_counts_conditioning_as_the_first_writer() {
        let conditioning = PrefixCacheConditioningEvidence {
            url: String::new(),
            model: String::new(),
            prompt_path: std::path::PathBuf::new(),
            prompt_sha256: String::new(),
            prompt_tokens: 8192,
            prompt: crate::workload::domain::ResolvedBenchPrompt::from_definition(
                &crate::workspace::BenchPrompt::Flat,
            ),
            request_body: BTreeMap::new(),
            maximum_shared_prefix_tokens: Some(8192),
            output_tokens: 1,
            consumes_population_entry: false,
            attention_data_parallel_size: 1,
            ranks: Vec::new(),
            succeeded: true,
            elapsed_ms: 1,
            error: None,
            completed_unix_ns: Some(0),
            group: 0,
        };
        let observations = [observation(0, 8064, 1_000, 100)];

        let evidence = prompt_reuse(
            BenchCacheStart::Primed,
            &observations,
            vec![&conditioning],
            1_000,
        );

        assert_eq!(
            evidence.map(|evidence| evidence.requests[0].opportunity),
            Some(ReuseOpportunity::Available)
        );
    }

    /// Requests of different prefix groups share no prefix, so a later
    /// request of another group is still the first of its own.
    #[test]
    fn only_writers_of_the_same_group_count() {
        let mut other_group = observation(1, 0, 2_000, 500);
        other_group.prefix_group = Some(1);
        let observations = [observation(0, 0, 0, 500), other_group];

        let opportunities = prompt_reuse(BenchCacheStart::Cold, &observations, Vec::new(), 1_000)
            .map(|evidence| {
                evidence
                    .requests
                    .iter()
                    .map(|request| request.opportunity)
                    .collect::<Vec<_>>()
            });

        assert_eq!(
            opportunities,
            Some(vec![ReuseOpportunity::First, ReuseOpportunity::First])
        );
    }

    #[test]
    fn an_uncontrolled_start_records_no_reuse_opportunity() {
        assert!(
            prompt_reuse(
                BenchCacheStart::Uncontrolled,
                &[observation(0, 0, 0, 1)],
                Vec::new(),
                0
            )
            .is_none()
        );
    }
}
