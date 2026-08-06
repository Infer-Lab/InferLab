//! Runtime-image and immutable external-image definition validation.

use super::{invalid, require_id, require_nonempty, require_reference};
use crate::InferlabError;
use crate::workspace::definitions::WorkspaceConfig;
use crate::workspace::source::is_safe_relative;
use std::collections::BTreeSet;

pub(super) fn validate(config: &WorkspaceConfig) -> Result<(), InferlabError> {
    for (id, image) in &config.images {
        require_id("image", id)?;
        require_reference("stack", &image.stack, &config.stacks)?;
        require_nonempty("base image", id, &image.base_image)?;
        if image.base_image.chars().any(char::is_whitespace) {
            return invalid(format!(
                "image {id:?} base image {:?} must not contain whitespace",
                image.base_image
            ));
        }
        if image.platforms.is_empty() {
            return invalid(format!("image {id:?} must declare at least one platform"));
        }
        let mut platforms = BTreeSet::new();
        for platform in &image.platforms {
            let mut parts = platform.split('/');
            let valid = matches!(
                (parts.next(), parts.next(), parts.next()),
                (Some(os), Some(arch), None) if !os.is_empty() && !arch.is_empty()
            );
            if !valid {
                return invalid(format!(
                    "image {id:?} platform {platform:?} must use the os/arch form"
                ));
            }
            if !platforms.insert(platform) {
                return invalid(format!(
                    "image {id:?} declares duplicate platform {platform:?}"
                ));
            }
        }
        if let Some(packages) = &image.packages {
            let stack = &config.stacks[&image.stack];
            for package in packages {
                if !is_safe_relative(package) {
                    return invalid(format!(
                        "image {id:?} package path {} must be workspace-relative without parent \
                         traversal",
                        package.display()
                    ));
                }
                if !stack.source_paths.contains(package) {
                    return invalid(format!(
                        "image {id:?} package path {} is not one of stack {:?}'s source_paths",
                        package.display(),
                        image.stack
                    ));
                }
            }
        }
        for coordinate in &image.validations {
            let Some(recipe) = config.recipes.get(&coordinate.recipe) else {
                return invalid(format!("unknown recipe {:?}", coordinate.recipe));
            };
            let server = &config.servers[&recipe.server];
            if let Some(case) = &coordinate.server_case
                && !server.cases.contains_key(case)
            {
                return invalid(format!(
                    "image {id:?} validation references unknown server case {case:?} of recipe {:?}",
                    coordinate.recipe,
                ));
            }
            if server.stack != image.stack {
                return invalid(format!(
                    "image {id:?} selects stack {:?} but validation recipe {:?} selects server stack {:?}; \
                     a validation recipe must run the serving stack the image contains",
                    image.stack, coordinate.recipe, server.stack
                ));
            }
        }
    }
    for (id, external) in &config.external_images {
        require_id("external image", id)?;
        require_nonempty("external image reference", id, &external.reference)?;
        if external.reference.chars().any(char::is_whitespace) {
            return invalid(format!(
                "external image {id:?} reference {:?} must not contain whitespace",
                external.reference
            ));
        }
        // Digest pinning makes a committed baseline mean one artifact
        // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        let digest_pinned =
            external
                .reference
                .rsplit_once("@sha256:")
                .is_some_and(|(repository, digest)| {
                    !repository.is_empty()
                        && digest.len() == 64
                        && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                });
        if !digest_pinned {
            return invalid(format!(
                "external image {id:?} reference {:?} must carry its immutable digest \
                 (repository[:tag]@sha256:<64 hex>)",
                external.reference
            ));
        }
        if external.integration.is_empty()
            || !external
                .integration
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        {
            return invalid(format!(
                "external image {id:?} claims invalid integration identifier {:?}",
                external.integration
            ));
        }
        // The integration package's presence in the committed dependency set
        // is verified against the parsed Pixi manifest in `validate_pixi`
        // ([[RFC-0006:C-INTEGRATIONS]]).
    }
    Ok(())
}
