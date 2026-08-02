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
    merge_exact(definition, patch, "")
}

fn invalid_override(raw_override: &str, message: String) -> InferlabError {
    InferlabError::InvalidOverride {
        value: raw_override.to_owned(),
        message,
    }
}

fn merge_exact(current: &mut toml::Value, patch: toml::Value, parent: &str) -> Result<(), String> {
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
                        merge_exact(existing, value, &path)?;
                    }
                    Some(existing) if !existing.is_table() && value.is_table() => {
                        return Err(format!("override traverses non-table value at {path}"));
                    }
                    _ => {
                        current.insert(key, value);
                    }
                }
            }
            Ok(())
        }
        _ => Err(format!("override traverses non-table value at {parent}")),
    }
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
}
