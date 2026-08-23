use super::super::FactSection;
use crate::workspace::{
    BenchCacheStart, BenchDefinition, BenchPrefixSharing, BenchPrompt, BenchPromptSelection,
    BenchRequestSource, BenchSessionSource, BenchSharedSystemContent, BenchTokenSelector,
    RequestRate,
};

pub(in crate::tui) struct DefinitionDetail {
    pub(in crate::tui) relationship: String,
    pub(in crate::tui) sections: Vec<FactSection>,
}

pub(in crate::tui) fn definition(definition: &BenchDefinition) -> DefinitionDetail {
    match definition {
        BenchDefinition::Serving {
            request_source,
            session_source,
            agentic_source,
            seed,
            concurrency,
            prompts_per_concurrency,
            warmup_prompts_per_concurrency,
            sessions_per_concurrency,
            warmup_sessions_per_concurrency,
            request_rates,
            request_count,
            duration_seconds,
            burstiness,
            cache,
            timeout_seconds,
            ..
        } => {
            let (relationship, source) = source_section(
                request_source.as_ref(),
                session_source.as_ref(),
                agentic_source
                    .as_ref()
                    .map(|source| (source.dataset.as_str(), source.profile.as_str())),
            );
            let mut sections = vec![source];
            if request_source.is_some() {
                let mut population = Vec::new();
                if let Some(value) = prompts_per_concurrency {
                    population.push(fact("Prompts / concurrency", value.to_string()));
                }
                population.push(fact(
                    "Warmup prompts / concurrency",
                    warmup_prompts_per_concurrency.to_string(),
                ));
                sections.push(FactSection {
                    title: "POPULATION",
                    rows: population,
                });
                let mut load = vec![fact("Concurrency", number_list(concurrency))];
                if !request_rates.is_empty() {
                    load.push(fact("Request rates", request_rate_list(request_rates)));
                }
                if let Some(value) = request_count {
                    load.push(fact("Request count", value.to_string()));
                }
                if let Some(value) = duration_seconds {
                    load.push(fact("Duration", format!("{value}s")));
                }
                if let Some(value) = burstiness {
                    load.push(fact("Burstiness", value.to_string()));
                }
                load.extend([
                    fact("Seed", seed.to_string()),
                    fact(
                        "Cache start",
                        cache_start(cache.as_ref().map(|cache| cache.start)),
                    ),
                    fact("Timeout", format!("{timeout_seconds}s")),
                ]);
                sections.push(FactSection {
                    title: "LOAD",
                    rows: load,
                });
            } else if session_source.is_some() {
                let mut population = Vec::new();
                if let Some(value) = sessions_per_concurrency {
                    population.push(fact("Sessions / concurrency", value.to_string()));
                }
                population.push(fact(
                    "Warmup sessions / concurrency",
                    warmup_sessions_per_concurrency.to_string(),
                ));
                sections.push(FactSection {
                    title: "POPULATION",
                    rows: population,
                });
                sections.push(FactSection {
                    title: "LOAD",
                    rows: vec![
                        fact("Concurrency", number_list(concurrency)),
                        fact("Seed", seed.to_string()),
                        fact(
                            "Cache start",
                            cache_start(cache.as_ref().map(|cache| cache.start)),
                        ),
                        fact("Timeout", format!("{timeout_seconds}s")),
                    ],
                });
            } else if agentic_source.is_some() {
                let mut load = vec![
                    fact("Root-tree concurrency", number_list(concurrency)),
                    fact("Seed", seed.to_string()),
                    fact("Timeout", format!("{timeout_seconds}s")),
                ];
                if let Some(value) = duration_seconds {
                    load.insert(1, fact("Profiling duration", format!("{value}s")));
                }
                sections.push(FactSection {
                    title: "LOAD",
                    rows: load,
                });
            }
            DefinitionDetail {
                relationship,
                sections,
            }
        }
        BenchDefinition::AdaptiveServing {
            request_source,
            seed,
            initial_request_rates,
            max_search_steps,
            min_rate_resolution,
            request_count,
            duration_seconds,
            burstiness,
            cache,
            timeout_seconds,
            ..
        } => {
            let (relationship, source) = source_section(Some(request_source), None, None);
            let mut load = vec![
                fact("Initial request rates", float_list(initial_request_rates)),
                fact("Maximum search steps", max_search_steps.to_string()),
            ];
            if let Some(value) = min_rate_resolution {
                load.push(fact("Minimum rate resolution", value.to_string()));
            }
            if let Some(value) = request_count {
                load.push(fact("Request count", value.to_string()));
            }
            if let Some(value) = duration_seconds {
                load.push(fact("Duration", format!("{value}s")));
            }
            if let Some(value) = burstiness {
                load.push(fact("Burstiness", value.to_string()));
            }
            load.extend([
                fact("Seed", seed.to_string()),
                fact(
                    "Cache start",
                    cache_start(cache.as_ref().map(|cache| cache.start)),
                ),
                fact("Timeout", format!("{timeout_seconds}s")),
            ]);
            DefinitionDetail {
                relationship: format!("adaptive {relationship}"),
                sections: vec![
                    source,
                    FactSection {
                        title: "LOAD",
                        rows: load,
                    },
                ],
            }
        }
    }
}

fn cache_start(start: Option<BenchCacheStart>) -> &'static str {
    match start.unwrap_or_default() {
        BenchCacheStart::Uncontrolled => "uncontrolled (default)",
        BenchCacheStart::Cold => "cold",
        BenchCacheStart::Primed => "primed",
    }
}

fn source_section(
    request: Option<&BenchRequestSource>,
    session: Option<&BenchSessionSource>,
    agentic: Option<(&str, &str)>,
) -> (String, FactSection) {
    if let Some(source) = request {
        return request_source_section(source);
    }
    if let Some(source) = session {
        let mut rows = vec![fact("Dataset", source.dataset.clone())];
        if let Some(profile) = source.profile.as_deref() {
            rows.push(fact("Profile", profile));
        }
        rows.extend([
            fact("Prompt", "server chat (source-owned)"),
            fact("Maximum input", format!("{} tok", source.max_input_tokens)),
            fact("Output limit", optional_tokens(source.output_tokens)),
            fact(
                "Inter-turn delay scale",
                source.inter_turn_delay_scale.to_string(),
            ),
        ]);
        if let Some(value) = source.max_inter_turn_delay_seconds {
            rows.push(fact("Maximum inter-turn delay", format!("{value}s")));
        }
        return (
            format!("linear session · dataset {}", source.dataset),
            FactSection {
                title: "SOURCE · LINEAR SESSION",
                rows,
            },
        );
    }
    if let Some((dataset, profile)) = agentic {
        return (
            format!("agentic replay · {dataset}/{profile}"),
            FactSection {
                title: "SOURCE · AGENTIC REPLAY",
                rows: vec![fact("Dataset", dataset), fact("Profile", profile)],
            },
        );
    }
    (
        "source not declared".to_owned(),
        FactSection {
            title: "SOURCE",
            rows: vec![fact("Source", "not declared")],
        },
    )
}

fn request_source_section(source: &BenchRequestSource) -> (String, FactSection) {
    match source {
        BenchRequestSource::Random {
            prompt,
            input_tokens,
            output_tokens,
            prefix_sharing,
            shared_system_content,
            corpus,
        } => (
            "requests · random".to_owned(),
            FactSection {
                title: "SOURCE · REQUESTS",
                rows: vec![
                    fact("Generator", "random"),
                    fact("Prompt", prompt_summary(prompt)),
                    fact("Input tokens", token_selector(input_tokens)),
                    fact("Output tokens", token_selector(output_tokens)),
                    fact("Prefix sharing", prefix_summary(prefix_sharing.as_ref())),
                    fact(
                        "Shared system content",
                        shared_system_summary(shared_system_content.as_ref()),
                    ),
                    fact(
                        "Corpus",
                        corpus
                            .as_ref()
                            .map_or_else(|| "synthetic".to_owned(), |corpus| corpus.path.clone()),
                    ),
                ],
            },
        ),
        BenchRequestSource::RandomMixture {
            prompt,
            shapes,
            prefix_sharing,
        } => (
            "requests · random mixture".to_owned(),
            FactSection {
                title: "SOURCE · REQUESTS",
                rows: vec![
                    fact("Generator", "random mixture"),
                    fact("Prompt", prompt_summary(prompt)),
                    fact(
                        "Shapes",
                        shapes
                            .iter()
                            .map(|shape| {
                                format!(
                                    "{}→{} tok ×{}",
                                    shape.input_tokens, shape.output_tokens, shape.weight
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", "),
                    ),
                    fact("Prefix sharing", prefix_summary(prefix_sharing.as_ref())),
                ],
            },
        ),
        BenchRequestSource::Dataset {
            dataset,
            profile,
            max_input_tokens,
            output_tokens,
        } => {
            let mut rows = vec![fact("Generator", "dataset"), fact("Dataset", dataset)];
            if let Some(profile) = profile.as_deref() {
                rows.push(fact("Profile", profile));
            }
            rows.extend([
                fact("Prompt", "server chat (source-owned)"),
                fact("Maximum input", format!("{max_input_tokens} tok")),
                fact("Output limit", optional_tokens(*output_tokens)),
            ]);
            (
                format!("requests · dataset {dataset}"),
                FactSection {
                    title: "SOURCE · REQUESTS",
                    rows,
                },
            )
        }
        BenchRequestSource::Replay {
            path,
            expected_sha256,
            prompt,
            prefix_sharing,
        } => (
            format!("requests · replay {path}"),
            FactSection {
                title: "SOURCE · REQUESTS",
                rows: vec![
                    fact("Generator", "replay"),
                    fact("Path", path.clone()),
                    fact(
                        "Expected SHA-256",
                        optional_text(expected_sha256.as_deref()),
                    ),
                    fact("Prompt", prompt_summary(prompt)),
                    fact("Prefix sharing", prefix_summary(prefix_sharing.as_ref())),
                ],
            },
        ),
    }
}

fn prompt_summary(prompt: &BenchPromptSelection) -> String {
    let provenance = if prompt.declared().is_some() {
        "declared"
    } else {
        "default"
    };
    match prompt.effective() {
        BenchPrompt::Flat => format!("flat ({provenance})"),
        BenchPrompt::RenderedChat { chat_template, .. } => format!(
            "rendered chat · {} ({provenance})",
            chat_template.as_deref().unwrap_or("tokenizer default")
        ),
        BenchPrompt::ServerChat => format!("server chat ({provenance})"),
    }
}
pub(super) fn token_selector(selector: &BenchTokenSelector) -> String {
    match selector {
        BenchTokenSelector::Fixed(value) => format!("{value} tok"),
        BenchTokenSelector::InclusiveUniform { min, max } => {
            format!("uniform [{min}, {max}] tok")
        }
    }
}
pub(super) fn prefix_summary(prefix: Option<&BenchPrefixSharing>) -> String {
    match prefix {
        Some(BenchPrefixSharing::Tokens {
            shared_prefix_tokens,
        }) => format!("{shared_prefix_tokens} tok"),
        Some(BenchPrefixSharing::Ratio {
            shared_prefix_ratio,
        }) => format!("{}%", shared_prefix_ratio * 100.0),
        None => "none".to_owned(),
    }
}
pub(super) fn shared_system_summary(content: Option<&BenchSharedSystemContent>) -> String {
    match content {
        Some(BenchSharedSystemContent::Tokens { tokens }) => format!("{tokens} tok"),
        Some(BenchSharedSystemContent::Ratio { ratio }) => format!("{}%", ratio * 100.0),
        None => "none".to_owned(),
    }
}
pub(super) fn fact(label: impl Into<String>, value: impl Into<String>) -> (String, String) {
    (label.into(), value.into())
}
pub(super) fn optional_text(value: Option<&str>) -> String {
    value.unwrap_or("—").to_owned()
}

fn optional_tokens(value: Option<u32>) -> String {
    value.map_or_else(
        || "dataset target".to_owned(),
        |value| format!("{value} tok"),
    )
}

fn number_list(values: &[u32]) -> String {
    if values.is_empty() {
        "—".to_owned()
    } else {
        values
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn float_list(values: &[f64]) -> String {
    if values.is_empty() {
        "—".to_owned()
    } else {
        values
            .iter()
            .map(f64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn request_rate_list(values: &[RequestRate]) -> String {
    if values.is_empty() {
        return "—".to_owned();
    }
    values
        .iter()
        .map(|value| match value {
            RequestRate::Finite(value) => value.to_string(),
            RequestRate::Unbounded => "unbounded".to_owned(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) const fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}
