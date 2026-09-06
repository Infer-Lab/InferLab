//! The registry of built-in proxy implementations
//! ([[RFC-0006:C-INTEGRATIONS]]): one entry per control-plane-rendered
//! frontend, binding the plan-declaration wire name, the internal CLI
//! subcommand spelling, the proxy-kind behavior identity, and the
//! proxy-owned version and route facts to a single table so consumers never
//! restate them.

use crate::{sglang, trtllm, vllm_mooncake, vllm_nixl};

/// Behavioral identity of a built-in proxy implementation. The control plane
/// selects per-proxy CLI argument shapes from the kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuiltinProxyKind {
    VllmMooncake,
    VllmNixl,
    Sglang,
    Trtllm,
}

/// One built-in proxy implementation's identity and proxy-owned facts.
pub struct BuiltinProxySpec {
    pub kind: BuiltinProxyKind,
    /// The plan-declaration vocabulary: the frontend `implementation` value an
    /// adapter returns on the wire.
    pub wire_name: &'static str,
    pub command_name: &'static str,
    /// Implementation version the proxy owns; the control plane rejects
    /// adapter declarations whose `implementation_version` disagrees.
    pub version: u32,
    pub healthcheck_path: &'static str,
    pub reset_prefix_cache_path: Option<&'static str>,
    pub prime_prefix_cache_path: Option<&'static str>,
    pub completions_path: &'static str,
    pub chat_completions_path: &'static str,
}

pub const VLLM_MOONCAKE: BuiltinProxySpec = BuiltinProxySpec {
    kind: BuiltinProxyKind::VllmMooncake,
    wire_name: "vllm_mooncake",
    command_name: "vllm-mooncake",
    version: vllm_mooncake::VERSION,
    healthcheck_path: vllm_mooncake::HEALTHCHECK_PATH,
    reset_prefix_cache_path: Some(vllm_mooncake::RESET_PREFIX_CACHE_PATH),
    prime_prefix_cache_path: Some(vllm_mooncake::PRIME_PREFIX_CACHE_PATH),
    completions_path: vllm_mooncake::COMPLETIONS_PATH,
    chat_completions_path: vllm_mooncake::CHAT_COMPLETIONS_PATH,
};

pub const VLLM_NIXL: BuiltinProxySpec = BuiltinProxySpec {
    kind: BuiltinProxyKind::VllmNixl,
    wire_name: "vllm_nixl",
    command_name: "vllm-nixl",
    version: vllm_nixl::VERSION,
    healthcheck_path: vllm_nixl::HEALTHCHECK_PATH,
    reset_prefix_cache_path: Some(vllm_nixl::RESET_PREFIX_CACHE_PATH),
    prime_prefix_cache_path: Some(vllm_nixl::PRIME_PREFIX_CACHE_PATH),
    completions_path: vllm_nixl::COMPLETIONS_PATH,
    chat_completions_path: vllm_nixl::CHAT_COMPLETIONS_PATH,
};

pub const SGLANG: BuiltinProxySpec = BuiltinProxySpec {
    kind: BuiltinProxyKind::Sglang,
    wire_name: "sglang",
    command_name: "sglang",
    version: sglang::VERSION,
    healthcheck_path: sglang::HEALTHCHECK_PATH,
    reset_prefix_cache_path: Some(sglang::RESET_PREFIX_CACHE_PATH),
    prime_prefix_cache_path: Some(sglang::PRIME_PREFIX_CACHE_PATH),
    completions_path: sglang::COMPLETIONS_PATH,
    chat_completions_path: sglang::CHAT_COMPLETIONS_PATH,
};

pub const TRTLLM: BuiltinProxySpec = BuiltinProxySpec {
    kind: BuiltinProxyKind::Trtllm,
    wire_name: "trtllm",
    command_name: "trtllm",
    version: trtllm::VERSION,
    healthcheck_path: trtllm::HEALTHCHECK_PATH,
    reset_prefix_cache_path: None,
    prime_prefix_cache_path: None,
    completions_path: trtllm::COMPLETIONS_PATH,
    chat_completions_path: trtllm::CHAT_COMPLETIONS_PATH,
};

pub const BUILTIN_PROXIES: &[BuiltinProxySpec] = &[VLLM_MOONCAKE, VLLM_NIXL, SGLANG, TRTLLM];

pub fn builtin_proxy_by_wire_name(name: &str) -> Option<&'static BuiltinProxySpec> {
    BUILTIN_PROXIES.iter().find(|spec| spec.wire_name == name)
}
