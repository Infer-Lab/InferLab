use crate::InferlabError;

/// One ordered `--set PATH=<TOML-value>` declaration from an invocation.
///
/// CLI entry points parse this stream once. Product domains then select the
/// declarations addressed to one definition and apply them through the same
/// exact TOML assignment semantics.
#[derive(Clone)]
pub(crate) struct InvocationOverride {
    index: usize,
    raw: String,
    path: String,
    raw_value: String,
}

impl InvocationOverride {
    pub(crate) fn parse_all(overrides: &[String]) -> Result<Vec<Self>, InferlabError> {
        overrides
            .iter()
            .enumerate()
            .map(|(index, raw)| {
                let (path, raw_value) =
                    raw.split_once('=')
                        .ok_or_else(|| InferlabError::InvalidOverride {
                            value: raw.clone(),
                            message: "expected PATH=<TOML-value>".to_owned(),
                        })?;
                Ok(Self {
                    index,
                    raw: raw.clone(),
                    path: path.to_owned(),
                    raw_value: raw_value.to_owned(),
                })
            })
            .collect()
    }

    pub(crate) const fn index(&self) -> usize {
        self.index
    }

    pub(crate) fn raw(&self) -> &str {
        &self.raw
    }

    pub(crate) fn path(&self) -> &str {
        &self.path
    }

    pub(crate) fn under(&self, prefix: &str) -> Option<Self> {
        self.path.strip_prefix(prefix).map(|path| Self {
            index: self.index,
            raw: self.raw.clone(),
            path: path.to_owned(),
            raw_value: self.raw_value.clone(),
        })
    }

    pub(crate) fn assignment(&self) -> Result<ExactTomlOverride, InferlabError> {
        ExactTomlOverride::parse(&self.path, &self.raw_value, &self.raw)
    }
}

/// One invocation override parsed by the TOML implementation as an exact
/// key-path assignment. Product callers remain responsible for selecting the
/// definition the assignment may affect and for deserializing the result back
/// into that definition's closed Rust type.
#[derive(Clone)]
pub(crate) struct ExactTomlOverride {
    root_key: String,
    patch: toml::Value,
}

impl ExactTomlOverride {
    pub(crate) fn parse(
        path: &str,
        raw_value: &str,
        raw_override: &str,
    ) -> Result<Self, InferlabError> {
        if path.is_empty() || path.bytes().any(|byte| matches!(byte, b'\n' | b'\r')) {
            return Err(invalid_override(
                raw_override,
                "setting path must be one TOML key path".to_owned(),
            ));
        }

        // Parse the value independently so a newline cannot smuggle a second
        // assignment into the combined document.
        let value_document: toml::Table =
            toml::from_str(&format!("value = {raw_value}")).map_err(|error| {
                invalid_override(raw_override, format!("invalid TOML value: {error}"))
            })?;
        if value_document.len() != 1 || !value_document.contains_key("value") {
            return Err(invalid_override(
                raw_override,
                "override must contain exactly one TOML value".to_owned(),
            ));
        }

        // Let the TOML parser, rather than an Inferlab path parser, own dotted
        // and quoted-key semantics. The sentinel parse classifies path errors
        // separately from value errors before the actual assignment is parsed.
        let path_document: toml::Table =
            toml::from_str(&format!("{path} = 0")).map_err(|error| {
                invalid_override(raw_override, format!("invalid TOML key path: {error}"))
            })?;
        let root_key = path_document.keys().next().cloned().ok_or_else(|| {
            invalid_override(raw_override, "setting path must not be empty".to_owned())
        })?;

        let patch = toml::from_str::<toml::Table>(&format!("{path} = {raw_value}"))
            .map(toml::Value::Table)
            .map_err(|error| {
                invalid_override(
                    raw_override,
                    format!("invalid TOML override assignment: {error}"),
                )
            })?;
        Ok(Self { root_key, patch })
    }

    pub(crate) fn into_patch(self) -> toml::Value {
        self.patch
    }

    pub(crate) fn root_key(&self) -> &str {
        &self.root_key
    }

    pub(crate) fn apply_to(
        self,
        definition: &mut toml::Value,
        raw_override: &str,
    ) -> Result<(), InferlabError> {
        apply_toml_patch(definition, self.patch).map_err(|message| InferlabError::InvalidOverride {
            value: raw_override.to_owned(),
            message,
        })
    }
}

pub(crate) fn apply_toml_patch(
    definition: &mut toml::Value,
    patch: toml::Value,
) -> Result<(), String> {
    merge_exact(definition, patch, "", false)
}

/// Compose two framework `settings` maps directly: the patch table's keys are
/// setting names, so `extra_args` segmentation and per-flag group merge apply
/// at the top level ([[RFC-0003:C-RESOLUTION]]).
pub(crate) fn apply_settings_patch(
    settings: &mut toml::Value,
    patch: toml::Value,
) -> Result<(), String> {
    merge_exact(settings, patch, "settings", true)
}

fn invalid_override(raw_override: &str, message: String) -> InferlabError {
    InferlabError::InvalidOverride {
        value: raw_override.to_owned(),
        message,
    }
}

fn merge_exact(
    current: &mut toml::Value,
    patch: toml::Value,
    parent: &str,
    in_settings: bool,
) -> Result<(), String> {
    match (current, patch) {
        (toml::Value::Table(current), toml::Value::Table(patch)) => {
            for (key, value) in patch {
                let path = if parent.is_empty() {
                    key.clone()
                } else {
                    format!("{parent}.{key}")
                };
                match current.get_mut(&key) {
                    Some(existing) if existing.is_table() && value.is_table() => {
                        // A `settings` table is the framework settings map;
                        // only its direct entries carry extra_args semantics.
                        merge_exact(existing, value, &path, key == "settings")?;
                    }
                    Some(existing) if !existing.is_table() && value.is_table() => {
                        return Err(format!("override traverses non-table value at {path}"));
                    }
                    // [[RFC-0003:C-RESOLUTION]] extra_args composes by per-flag
                    // group last-wins merge; every other array keeps wholesale
                    // replacement.
                    Some(existing)
                        if key == "extra_args"
                            && in_settings
                            && existing.is_array()
                            && value.is_array() =>
                    {
                        merge_extra_args(existing, value, &path)?;
                    }
                    _ => {
                        // Fresh extra_args arrays skip the merge but still owe
                        // segmentation validation (e.g. a leading value token).
                        if key == "extra_args" && in_settings && value.is_array() {
                            segment_extra_args(&extra_args_tokens(&value, &path)?, &path)?;
                        }
                        current.insert(key, value);
                    }
                }
            }
            Ok(())
        }
        _ => Err(format!("override traverses non-table value at {parent}")),
    }
}

/// One segmented flag group: the flag token (or the verbatim `--` passthrough
/// block) plus every value token belonging to it.
struct ExtraArgsGroup {
    name: String,
    tokens: Vec<String>,
}

/// Segment one extra_args array into flag groups. A `--`-prefixed token opens
/// a group named by the flag text before any `=`; repeated names coalesce into
/// the first group's position so a repeatable flag forms one logical group per
/// layer. A bare `--` and every token after it form the single verbatim
/// passthrough block ([[RFC-0003:C-RESOLUTION]]): no named group may open
/// behind it, and a second bare `--` is malformed.
fn segment_extra_args(tokens: &[String], path: &str) -> Result<Vec<ExtraArgsGroup>, String> {
    let mut groups: Vec<ExtraArgsGroup> = Vec::new();
    for token in tokens {
        if let Some(verbatim) = groups.iter_mut().find(|group| group.name == "--") {
            if token == "--" {
                return Err(format!(
                    "extra_args at {path} declares a second bare `--`; one array carries a single verbatim passthrough block"
                ));
            }
            verbatim.tokens.push(token.clone());
            continue;
        }
        if token == "--" {
            groups.push(ExtraArgsGroup {
                name: "--".to_owned(),
                tokens: vec![token.clone()],
            });
            continue;
        }
        if token.starts_with("--") {
            let name = token.split('=').next().unwrap_or(token.as_str()).to_owned();
            if let Some(existing) = groups.iter_mut().find(|group| group.name == name) {
                existing.tokens.push(token.clone());
            } else {
                groups.push(ExtraArgsGroup {
                    name,
                    tokens: vec![token.clone()],
                });
            }
            continue;
        }
        match groups.last_mut() {
            Some(group) => group.tokens.push(token.clone()),
            None => {
                return Err(format!(
                    "extra_args value token {token:?} at {path} precedes any flag"
                ));
            }
        }
    }
    Ok(groups)
}

/// [[RFC-0003:C-RESOLUTION]] Validate one extra_args array's segmentation at
/// workspace load, before any layer composition runs: every value token
/// follows a flag, and one array carries at most one verbatim `--` block.
pub(crate) fn validate_extra_args_segmentation(
    tokens: &[String],
    path: &str,
) -> Result<(), String> {
    segment_extra_args(tokens, path).map(|_| ())
}

/// [[RFC-0003:C-RESOLUTION]] Merge the patch layer's groups into the base:
/// same-named groups replace in place, patch-only groups append in order, and
/// groups absent from the patch are inherited unchanged.
fn merge_extra_args(base: &mut toml::Value, patch: toml::Value, path: &str) -> Result<(), String> {
    let base_tokens = extra_args_tokens(base, path)?;
    let patch_tokens = extra_args_tokens(&patch, path)?;
    let mut groups = segment_extra_args(&base_tokens, path)?;
    for patch_group in segment_extra_args(&patch_tokens, path)? {
        match groups
            .iter_mut()
            .find(|group| group.name == patch_group.name)
        {
            Some(existing) => *existing = patch_group,
            None => groups.push(patch_group),
        }
    }
    *base = toml::Value::Array(
        groups
            .into_iter()
            .flat_map(|group| group.tokens.into_iter().map(toml::Value::String))
            .collect(),
    );
    Ok(())
}

fn extra_args_tokens(value: &toml::Value, path: &str) -> Result<Vec<String>, String> {
    let toml::Value::Array(items) = value else {
        return Err(format!("extra_args at {path} must be an array"));
    };
    items
        .iter()
        .map(|item| match item {
            toml::Value::String(token) => Ok(token.clone()),
            _ => Err(format!("extra_args entries at {path} must be strings")),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{ExactTomlOverride, InvocationOverride};

    #[test]
    fn invocation_stream_is_parsed_once_before_target_selection() -> Result<(), String> {
        let raw = vec![
            "server.settings.eager=true".to_owned(),
            "benches.latency.concurrency=[1, 2]".to_owned(),
        ];
        let overrides = InvocationOverride::parse_all(&raw).map_err(|error| error.to_string())?;
        let bench = overrides[1]
            .under("benches.latency.")
            .ok_or_else(|| "bench override was not selected".to_owned())?;

        assert_eq!(bench.index(), 1);
        assert_eq!(bench.raw(), raw[1]);
        assert_eq!(bench.path(), "concurrency");
        Ok(())
    }

    #[test]
    fn toml_owns_quoted_paths_and_structured_values() -> Result<(), String> {
        let patch = ExactTomlOverride::parse(
            r#"settings."framework.option""#,
            r#"{ enabled = true, limits = [1, 2] }"#,
            r#"settings."framework.option"={ enabled = true, limits = [1, 2] }"#,
        )
        .map_err(|error| error.to_string())?;

        assert_eq!(patch.root_key(), "settings");
        assert_eq!(
            patch.into_patch()["settings"]["framework.option"]["enabled"].as_bool(),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn exact_merge_replaces_arrays_and_rejects_scalar_traversal() -> Result<(), String> {
        let mut definition: toml::Value =
            toml::from_str("values = [1, 2]\nscalar = 1").map_err(|error| error.to_string())?;
        ExactTomlOverride::parse("values", "[3]", "values=[3]")
            .map_err(|error| error.to_string())?
            .apply_to(&mut definition, "values=[3]")
            .map_err(|error| error.to_string())?;
        assert_eq!(definition["values"].as_array().map(Vec::len), Some(1));

        let error = ExactTomlOverride::parse("scalar.child", "2", "scalar.child=2")
            .map_err(|error| error.to_string())?
            .apply_to(&mut definition, "scalar.child=2")
            .map_err(|error| error.to_string());
        assert!(matches!(
            error,
            Err(ref message)
                if message.ends_with("override traverses non-table value at scalar")
        ));
        Ok(())
    }

    use super::apply_toml_patch;

    fn settings_with(extra_args: &str) -> Result<toml::Value, String> {
        toml::from_str(&format!(
            "values = [1]\n[settings]\nextra_args = {extra_args}"
        ))
        .map_err(|error| error.to_string())
    }

    fn composed_extra_args(base: toml::Value, patch: toml::Value) -> Result<Vec<String>, String> {
        let mut definition = base;
        apply_toml_patch(&mut definition, patch)?;
        let tokens: Vec<String> = definition["settings"]["extra_args"]
            .as_array()
            .ok_or_else(|| "extra_args missing after merge".to_owned())?
            .iter()
            .filter_map(|token| token.as_str().map(str::to_owned))
            .collect();
        Ok(tokens)
    }

    #[test]
    fn extra_args_merge_replaces_named_groups_and_inherits_the_rest() -> Result<(), String> {
        let base = settings_with(
            r#"["--language-model-only", "--max-num-seqs", "256", "--max-num-batched-tokens", "8192"]"#,
        )?;
        let patch = settings_with(r#"["--max-num-seqs", "512"]"#)?;

        let merged = composed_extra_args(base, patch)?;
        assert_eq!(
            merged,
            [
                "--language-model-only",
                "--max-num-seqs",
                "512",
                "--max-num-batched-tokens",
                "8192"
            ],
            "the named group is replaced in place and the rest are inherited"
        );
        Ok(())
    }

    #[test]
    fn extra_args_merge_treats_a_repeatable_flag_as_one_group() -> Result<(), String> {
        let base = settings_with(r#"["--arg-dup", "base", "--other", "1"]"#)?;
        let patch = settings_with(r#"["--arg-dup", "a", "--arg-dup", "b"]"#)?;

        let merged = composed_extra_args(base, patch)?;
        assert_eq!(merged, ["--arg-dup", "a", "--arg-dup", "b", "--other", "1"]);
        Ok(())
    }

    #[test]
    fn extra_args_merge_groups_equals_spelling_and_replaces_passthrough_wholesale()
    -> Result<(), String> {
        let base = settings_with(r#"["--max-num-seqs=256", "--", "--block-size", "32"]"#)?;
        let patch = settings_with(r#"["--max-num-seqs=512", "--", "--block-size", "64"]"#)?;

        let merged = composed_extra_args(base, patch)?;
        assert_eq!(merged, ["--max-num-seqs=512", "--", "--block-size", "64"]);
        Ok(())
    }

    #[test]
    fn extra_args_merge_inherits_the_passthrough_block_when_the_patch_has_none()
    -> Result<(), String> {
        let base = settings_with(r#"["--max-num-seqs", "256", "--", "--block-size", "32"]"#)?;
        let patch = settings_with(r#"["--max-num-seqs", "512"]"#)?;

        let merged = composed_extra_args(base, patch)?;
        assert_eq!(
            merged,
            ["--max-num-seqs", "512", "--", "--block-size", "32"]
        );
        Ok(())
    }

    #[test]
    fn extra_args_merge_replaces_the_whole_passthrough_block_not_single_flags() -> Result<(), String>
    {
        let base = settings_with(
            r#"["--max-num-seqs", "256", "--", "--block-size", "32", "--num-blocks", "8"]"#,
        )?;
        // The patch rewrites only one flag behind the sentinel; the base
        // block's other flag must not survive the replacement.
        let patch = settings_with(r#"["--", "--block-size", "64"]"#)?;

        let merged = composed_extra_args(base, patch)?;
        assert_eq!(
            merged,
            ["--max-num-seqs", "256", "--", "--block-size", "64"],
            "the verbatim passthrough block is replaced as a whole"
        );
        Ok(())
    }

    #[test]
    fn extra_args_rejects_a_second_verbatim_sentinel() -> Result<(), String> {
        let error = composed_extra_args(
            settings_with(r#"["--known", "1"]"#)?,
            settings_with(r#"["--", "--a", "--", "--b"]"#)?,
        );
        assert!(matches!(
            error,
            Err(ref message) if message.contains("second bare `--`")
        ));

        // A fresh extra_args array (no base entry) owes the same validation.
        let mut definition: toml::Value =
            toml::from_str("[settings]\nvalues = [1]").map_err(|error| error.to_string())?;
        let error = apply_toml_patch(
            &mut definition,
            toml::from_str::<toml::Value>("[settings]\nextra_args = [\"--\", \"--a\", \"--\"]")
                .map_err(|error| error.to_string())?,
        );
        assert!(matches!(
            error,
            Err(ref message) if message.contains("second bare `--`")
        ));
        Ok(())
    }

    #[test]
    fn extra_args_rejects_non_string_entries() -> Result<(), String> {
        let mut definition: toml::Value =
            toml::from_str("[settings]\nextra_args = [\"--a\", \"1\"]")
                .map_err(|error| error.to_string())?;
        let error = apply_toml_patch(
            &mut definition,
            toml::from_str::<toml::Value>("[settings]\nextra_args = [\"--b\", 2]")
                .map_err(|error| error.to_string())?,
        );
        assert!(matches!(
            error,
            Err(ref message) if message.contains("must be strings")
        ));
        Ok(())
    }

    #[test]
    fn extra_args_full_restate_cases_produce_the_identical_argv() -> Result<(), String> {
        let base_tokens = r#"["--max-num-seqs", "64", "--moe-backend", "flashinfer_cutlass", "--speculative-config", "{}"]"#;
        let base = settings_with(base_tokens)?;
        let patch = settings_with(
            r#"["--max-num-seqs", "64", "--moe-backend", "b12x", "--speculative-config", "{}"]"#,
        )?;

        let merged = composed_extra_args(base, patch)?;
        assert_eq!(
            merged,
            [
                "--max-num-seqs",
                "64",
                "--moe-backend",
                "b12x",
                "--speculative-config",
                "{}"
            ],
            "old-style restating cases keep their exact argv under merge semantics"
        );
        Ok(())
    }

    #[test]
    fn extra_args_rejects_a_value_token_before_any_flag() -> Result<(), String> {
        let base = settings_with(r#"["--known", "1"]"#)?;
        let error = composed_extra_args(base, settings_with(r#"["stray-value", "--known", "2"]"#)?);
        assert!(matches!(
            error,
            Err(ref message) if message.contains("precedes any flag")
        ));

        let error = composed_extra_args(
            settings_with(r#"["--known", "1"]"#)?,
            settings_with(r#"["stray-value"]"#)?,
        );
        assert!(matches!(
            error,
            Err(ref message) if message.contains("precedes any flag")
        ));

        // A fresh extra_args array (no base entry) owes the same validation.
        let mut definition: toml::Value =
            toml::from_str("[settings]\nvalues = [1]").map_err(|error| error.to_string())?;
        let error = apply_toml_patch(
            &mut definition,
            toml::from_str::<toml::Value>("[settings]\nextra_args = [\"stray-value\"]")
                .map_err(|error| error.to_string())?,
        );
        assert!(matches!(
            error,
            Err(ref message) if message.contains("precedes any flag")
        ));
        Ok(())
    }

    #[test]
    fn extra_args_merge_applies_under_role_settings_but_not_unrelated_paths() -> Result<(), String>
    {
        let mut definition: toml::Value = toml::from_str(
            "[roles.serve.settings]\nextra_args = [\"--a\", \"1\"]\n[other]\nextra_args = [\"--a\", \"1\"]",
        )
        .map_err(|error| error.to_string())?;
        let patch: toml::Value = toml::from_str(
            "[roles.serve.settings]\nextra_args = [\"--b\", \"2\"]\n[other]\nextra_args = [\"--b\", \"2\"]",
        )
        .map_err(|error| error.to_string())?;
        apply_toml_patch(&mut definition, patch)?;

        let role_tokens: Vec<&str> = definition["roles"]["serve"]["settings"]["extra_args"]
            .as_array()
            .ok_or_else(|| "role extra_args stays an array".to_owned())?
            .iter()
            .filter_map(|token| token.as_str())
            .collect();
        assert_eq!(role_tokens, ["--a", "1", "--b", "2"]);

        let other_tokens: Vec<&str> = definition["other"]["extra_args"]
            .as_array()
            .ok_or_else(|| "other extra_args stays an array".to_owned())?
            .iter()
            .filter_map(|token| token.as_str())
            .collect();
        assert_eq!(
            other_tokens,
            ["--b", "2"],
            "arrays outside settings keep wholesale replacement"
        );
        Ok(())
    }
}
