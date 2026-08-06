use super::classify_slo_evaluation;
use crate::bench_metric::{BenchMetric, DistributionFamily, DistributionStatistic};
use crate::workload::adaptive::ProbeClassification;
use crate::workload::record::{
    AggregateSloEvaluation, CaseSloEvaluation, SloBoundDirection, SloEvaluationOutcome,
};

#[test]
fn unavailable_constraint_does_not_erase_an_above_region_failure() {
    let evaluation = CaseSloEvaluation {
        aggregate_slos: vec![
            AggregateSloEvaluation {
                metric: BenchMetric::PromptCacheReadRatio,
                direction: SloBoundDirection::AtLeast,
                bound: 0.5,
                observed: None,
                outcome: SloEvaluationOutcome::Unavailable,
            },
            AggregateSloEvaluation {
                metric: BenchMetric::Distribution {
                    statistic: DistributionStatistic::P99,
                    family: DistributionFamily::Ttft,
                },
                direction: SloBoundDirection::AtMost,
                bound: 100.0,
                observed: Some(150.0),
                outcome: SloEvaluationOutcome::Failed,
            },
        ],
        request_slo: None,
        passed: false,
    };

    assert_eq!(
        classify_slo_evaluation(&evaluation),
        ProbeClassification::Above
    );

    let below_evaluation = CaseSloEvaluation {
        aggregate_slos: vec![
            AggregateSloEvaluation {
                metric: BenchMetric::PromptCacheReadRatio,
                direction: SloBoundDirection::AtLeast,
                bound: 0.5,
                observed: None,
                outcome: SloEvaluationOutcome::Unavailable,
            },
            AggregateSloEvaluation {
                metric: BenchMetric::RequestThroughput,
                direction: SloBoundDirection::AtLeast,
                bound: 10.0,
                observed: Some(5.0),
                outcome: SloEvaluationOutcome::Failed,
            },
        ],
        request_slo: None,
        passed: false,
    };
    assert_eq!(
        classify_slo_evaluation(&below_evaluation),
        ProbeClassification::Below
    );
}
