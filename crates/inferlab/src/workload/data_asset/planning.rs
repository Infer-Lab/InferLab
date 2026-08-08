//! Effect-free planning and sharing of measurement source preparation.

use super::model::{
    AgenticDataAssetSource, DataAssetCacheOutcome, DataAssetCacheStore, DataAssetConsumer,
    DataAssetConsumerKind, DataAssetDryRunProjection, DataAssetObservation, DataAssetPlan,
    DataAssetPlannedEffect, DataAssetPreparationAttempt, DataAssetSource, DataAssetUnavailableFact,
    EvalDataAssetSource, ReleaseCatalogDataAssetSource,
};
use crate::InferlabError;
use crate::workload::domain::{ResolvedBenchRequestSource, ResolvedBenchSource};
use crate::workload::plan::{BenchPlan, EvalExecutionPlan, EvalPlan};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

pub(crate) fn plan_measurement_data_assets(
    workspace_root: &Path,
    evals: &[EvalPlan],
    benches: &[BenchPlan],
) -> Result<Vec<DataAssetPlan>, InferlabError> {
    let mut plans = Vec::new();
    let mut by_key = BTreeMap::new();
    for eval in evals {
        let Some(source) = eval_asset(workspace_root, eval) else {
            continue;
        };
        add_plan(
            &mut plans,
            &mut by_key,
            source,
            DataAssetConsumer {
                kind: DataAssetConsumerKind::Eval,
                definition_id: eval.id.clone(),
            },
        )?;
    }
    for bench in benches {
        let Some(source) = bench_asset(bench) else {
            continue;
        };
        add_plan(
            &mut plans,
            &mut by_key,
            source,
            DataAssetConsumer {
                kind: DataAssetConsumerKind::Bench,
                definition_id: bench.id.clone(),
            },
        )?;
    }
    Ok(plans)
}

fn add_plan(
    plans: &mut Vec<DataAssetPlan>,
    by_key: &mut BTreeMap<String, usize>,
    source: DataAssetSource,
    consumer: DataAssetConsumer,
) -> Result<usize, InferlabError> {
    let bytes = source
        .key_bytes()
        .map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to encode measurement data-asset source key: {error}"),
        })?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if let Some(index) = by_key.get(&digest).copied() {
        plans[index].consumers.push(consumer);
        return Ok(index);
    }
    let dry_run = dry_run_projection(&source);
    by_key.insert(digest.clone(), plans.len());
    plans.push(DataAssetPlan {
        attempt_id: format!("data-asset-{}", &digest[..16]),
        source_key_sha256: digest,
        source,
        consumers: vec![consumer],
        dry_run,
    });
    Ok(plans.len() - 1)
}

fn eval_asset(workspace_root: &Path, plan: &EvalPlan) -> Option<DataAssetSource> {
    let EvalExecutionPlan::LmEval {
        toolchain,
        bundled_task,
        command,
    } = &plan.execution
    else {
        return None;
    };
    if !matches!(
        plan.definition,
        crate::workspace::EvalDefinition::LmEval { .. }
    ) {
        return None;
    }
    Some(DataAssetSource::Eval {
        source: Box::new(EvalDataAssetSource {
            workspace_root: workspace_root.to_path_buf(),
            workspace_source_exclusions: plan.workspace_source_exclusions.clone(),
            definition: Box::new(plan.definition.clone()),
            bundled_task: bundled_task.clone(),
            command: command.clone(),
            acquisition_runtime_identity: format!(
                "{}:{}:{}:{}",
                toolchain.runner_version,
                toolchain.runner_sha256,
                toolchain.lm_eval_version,
                toolchain.lock_sha256
            ),
        }),
    })
}

fn bench_asset(plan: &BenchPlan) -> Option<DataAssetSource> {
    let runtime = &plan.client.toolchain;
    let runtime_identity = format!(
        "{}:{}:{}:{}",
        runtime.runner_version, runtime.runner_sha256, runtime.aiperf_version, runtime.lock_sha256
    );
    match &plan.client.effective_definition.source {
        ResolvedBenchSource::Requests {
            request_source: ResolvedBenchRequestSource::Dataset { catalog, .. },
        } => Some(DataAssetSource::ReleaseCatalog {
            source: Box::new(ReleaseCatalogDataAssetSource {
                dataset: catalog.dataset.clone(),
                profile: catalog.profile.clone(),
                url: catalog.url.clone(),
                upstream_identity: catalog.upstream_identity.clone(),
                expected_sha256: catalog.sha256.clone(),
                configuration: catalog.configuration.clone(),
                split: catalog.split.clone(),
                filter: catalog.filter.clone(),
                cache_path: catalog.cache_path.clone(),
                acquisition_runtime_identity: format!("inferlab-{}", env!("CARGO_PKG_VERSION")),
            }),
        }),
        ResolvedBenchSource::Sessions { session_source } => {
            let catalog = &session_source.catalog;
            Some(DataAssetSource::ReleaseCatalog {
                source: Box::new(ReleaseCatalogDataAssetSource {
                    dataset: catalog.dataset.clone(),
                    profile: catalog.profile.clone(),
                    url: catalog.url.clone(),
                    upstream_identity: catalog.upstream_identity.clone(),
                    expected_sha256: catalog.sha256.clone(),
                    configuration: catalog.configuration.clone(),
                    split: catalog.split.clone(),
                    filter: catalog.filter.clone(),
                    cache_path: catalog.cache_path.clone(),
                    acquisition_runtime_identity: format!("inferlab-{}", env!("CARGO_PKG_VERSION")),
                }),
            })
        }
        ResolvedBenchSource::Agentic { agentic_source } => Some(DataAssetSource::Agentic {
            source: Box::new(AgenticDataAssetSource {
                command: plan.client.command.clone(),
                definition: Box::new(agentic_source.clone()),
                acquisition_runtime_identity: runtime_identity,
            }),
        }),
        ResolvedBenchSource::Requests { .. } => None,
    }
}

fn dry_run_projection(source: &DataAssetSource) -> DataAssetDryRunProjection {
    if let DataAssetSource::Eval { source } = source
        && let crate::workspace::EvalDefinition::LmEval {
            task: crate::workspace::EvalTaskSource::WorkspaceYaml { yaml: path },
            ..
        } = source.definition.as_ref()
    {
        return DataAssetDryRunProjection::LocalObservation {
            effective_selection: None,
            cache_stores: Vec::new(),
            observations: vec![if path.is_file() {
                DataAssetObservation::SelectedLocalPathPresent
            } else {
                DataAssetObservation::SelectedLocalPathMissing
            }],
            planned_external_work: vec![DataAssetPlannedEffect::OwningRuntimeSourceResolution],
            unavailable: vec![
                DataAssetUnavailableFact::CompleteLocalClosure,
                DataAssetUnavailableFact::PreparedSnapshotIdentity,
                DataAssetUnavailableFact::ReproducibilityConclusion,
            ],
        };
    }
    if let DataAssetSource::ReleaseCatalog { source } = source {
        return DataAssetDryRunProjection::LocalObservation {
            effective_selection: None,
            cache_stores: vec![DataAssetCacheStore {
                authority: "inferlab_http_cas".to_owned(),
                purpose: "release_catalog_source".to_owned(),
                path: Some(source.cache_path.clone()),
                outcome: if source.cache_path.is_file() {
                    DataAssetCacheOutcome::PartialReuse
                } else {
                    DataAssetCacheOutcome::Miss
                },
            }],
            observations: vec![if source.cache_path.is_file() {
                DataAssetObservation::CachePathPresent
            } else {
                DataAssetObservation::CachePathMissing
            }],
            planned_external_work: vec![
                DataAssetPlannedEffect::ReleaseSourceAcquisitionAndVerification,
            ],
            unavailable: vec![
                DataAssetUnavailableFact::DigestVerification,
                DataAssetUnavailableFact::AcquiredSource,
                DataAssetUnavailableFact::ReproducibilityConclusion,
            ],
        };
    }
    if let DataAssetSource::Eval { source } = source
        && let crate::workspace::EvalDefinition::LmEval {
            task: crate::workspace::EvalTaskSource::Bundled { .. },
            ..
        } = source.definition.as_ref()
    {
        return DataAssetDryRunProjection::LocalObservation {
            effective_selection: None,
            cache_stores: Vec::new(),
            observations: vec![DataAssetObservation::ReleaseBundledClosureSelected],
            planned_external_work: vec![DataAssetPlannedEffect::ReleaseAssetVerification],
            unavailable: vec![
                DataAssetUnavailableFact::DigestVerification,
                DataAssetUnavailableFact::ReproducibilityConclusion,
            ],
        };
    }
    DataAssetDryRunProjection::Planned {
        external_work: vec![DataAssetPlannedEffect::SourceResolutionOrAcquisition],
        unavailable: vec![
            DataAssetUnavailableFact::AcquiredSource,
            DataAssetUnavailableFact::ReproducibilityConclusion,
        ],
    }
}

pub(crate) fn attempts_from_plans(plans: &[DataAssetPlan]) -> Vec<DataAssetPreparationAttempt> {
    plans
        .iter()
        .map(DataAssetPreparationAttempt::from)
        .collect()
}

pub(crate) fn attempt_id_for(
    plans: &[DataAssetPlan],
    kind: DataAssetConsumerKind,
    definition_id: &str,
) -> Result<Option<String>, InferlabError> {
    let mut matches = plans
        .iter()
        .filter(|plan| {
            plan.consumers
                .iter()
                .any(|consumer| consumer.kind == kind && consumer.definition_id == definition_id)
        })
        .map(|plan| plan.attempt_id.clone());
    let first = matches.next();
    if matches.next().is_some() {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "measurement {definition_id:?} resolved more than one data-asset source"
            ),
        });
    }
    Ok(first)
}

#[cfg(test)]
mod tests {
    use super::{
        DataAssetConsumer, DataAssetConsumerKind, DataAssetDryRunProjection, DataAssetSource,
        add_plan, attempt_id_for, dry_run_projection,
    };
    use crate::workload::data_asset::model::{
        DataAssetObservation, DataAssetUnavailableFact, EvalDataAssetSource,
        ReleaseCatalogDataAssetSource,
    };
    use crate::workload::plan::ClientCommandPlan;
    use crate::workspace::{EvalDefinition, EvalTaskSource};
    use std::collections::BTreeMap;

    fn source(split: &str) -> DataAssetSource {
        DataAssetSource::ReleaseCatalog {
            source: Box::new(ReleaseCatalogDataAssetSource {
                dataset: "fixture".to_owned(),
                profile: Some("default".to_owned()),
                url: "https://example.invalid/data.json".to_owned(),
                upstream_identity: "revision".to_owned(),
                expected_sha256: "digest".to_owned(),
                configuration: Some("default".to_owned()),
                split: Some(split.to_owned()),
                filter: None,
                cache_path: "fixture-cache.json".into(),
                acquisition_runtime_identity: "inferlab-0.10.0".to_owned(),
            }),
        }
    }

    fn consumer(id: &str) -> DataAssetConsumer {
        DataAssetConsumer {
            kind: DataAssetConsumerKind::Bench,
            definition_id: id.to_owned(),
        }
    }

    #[test]
    fn equal_source_keys_share_one_attempt() -> Result<(), crate::InferlabError> {
        let mut plans = Vec::new();
        let mut by_key = BTreeMap::new();

        let first = add_plan(
            &mut plans,
            &mut by_key,
            source("test"),
            consumer("request-bench"),
        )?;
        let second = add_plan(
            &mut plans,
            &mut by_key,
            source("test"),
            consumer("session-bench"),
        )?;

        assert_eq!(first, second);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].consumers.len(), 2);
        assert_eq!(
            attempt_id_for(&plans, DataAssetConsumerKind::Bench, "request-bench")?,
            Some(plans[0].attempt_id.clone())
        );

        Ok(())
    }

    #[test]
    fn a_distinct_split_gets_a_distinct_attempt() -> Result<(), crate::InferlabError> {
        let mut plans = Vec::new();
        let mut by_key = BTreeMap::new();

        let first = add_plan(
            &mut plans,
            &mut by_key,
            source("test"),
            consumer("test-bench"),
        )?;
        let second = add_plan(
            &mut plans,
            &mut by_key,
            source("train"),
            consumer("train-bench"),
        )?;

        assert_ne!(first, second);
        assert_eq!(plans.len(), 2);
        Ok(())
    }

    #[test]
    fn a_distinct_filter_gets_a_distinct_attempt() -> Result<(), crate::InferlabError> {
        let mut plans = Vec::new();
        let mut by_key = BTreeMap::new();
        let mut first_source = source("test");
        let DataAssetSource::ReleaseCatalog { source: catalog } = &mut first_source else {
            return Err(crate::InferlabError::InvalidConfig {
                message: "fixture source is not release catalog".to_owned(),
            });
        };
        catalog.filter = Some(crate::workload::domain::BenchDatasetFilter {
            field: "language".to_owned(),
            value: "en".to_owned(),
        });
        let mut second_source = source("test");
        let DataAssetSource::ReleaseCatalog { source: catalog } = &mut second_source else {
            return Err(crate::InferlabError::InvalidConfig {
                message: "fixture source is not release catalog".to_owned(),
            });
        };
        catalog.filter = Some(crate::workload::domain::BenchDatasetFilter {
            field: "language".to_owned(),
            value: "fr".to_owned(),
        });

        let first = add_plan(&mut plans, &mut by_key, first_source, consumer("english"))?;
        let second = add_plan(&mut plans, &mut by_key, second_source, consumer("french"))?;

        assert_ne!(first, second);
        assert_eq!(plans.len(), 2);
        Ok(())
    }

    #[test]
    fn local_dry_run_observes_the_path_without_claiming_a_snapshot()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("task.yaml");
        std::fs::write(&path, "task: local\n")?;
        let source = DataAssetSource::Eval {
            source: Box::new(EvalDataAssetSource {
                workspace_root: root.path().to_path_buf(),
                workspace_source_exclusions: Vec::new(),
                definition: Box::new(EvalDefinition::LmEval {
                    task: EvalTaskSource::WorkspaceYaml { yaml: path },
                    prompt: Default::default(),
                    request_body: Default::default(),
                    limit: None,
                    few_shot: None,
                    seed: None,
                    trials: 1,
                    max_tokens: None,
                    concurrency: None,
                    metric: "acc".to_owned(),
                    metric_filter: None,
                    threshold: 0.0,
                    timeout_seconds: 60,
                }),
                bundled_task: None,
                command: ClientCommandPlan {
                    argv: Vec::new(),
                    env: Default::default(),
                    cwd: root.path().to_path_buf(),
                },
                acquisition_runtime_identity: "runner".to_owned(),
            }),
        };

        let DataAssetDryRunProjection::LocalObservation {
            observations,
            unavailable,
            ..
        } = dry_run_projection(&source)
        else {
            return Err("local source did not produce a local observation".into());
        };
        assert_eq!(
            observations,
            [DataAssetObservation::SelectedLocalPathPresent]
        );
        assert!(unavailable.contains(&DataAssetUnavailableFact::PreparedSnapshotIdentity));
        Ok(())
    }
}
