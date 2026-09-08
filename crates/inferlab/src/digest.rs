//! Content digests of local files.

use crate::InferlabError;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

pub(crate) fn hash_file(path: &Path) -> Result<String, InferlabError> {
    let bytes = fs::read(path).map_err(|source| InferlabError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

/// The release-owned recursive enumeration digest of an operator image
/// directory ([[RFC-0004:C-BENCH-REQUEST-SOURCES]]).
///
/// Composition rule: every regular file reachable under `directory` (symbolic
/// links and other non-regular entries are not followed) contributes its
/// relative path rendered with `/` separators and its content digest; the
/// entries are ordered by the byte sequence of that relative path, and the
/// directory digest is SHA-256 over, per file in order, the relative-path
/// bytes, one `0x00` separator, the file's 32-byte SHA-256, and one `0x00`
/// terminator. A directory that enumerates no regular file has no digest and
/// fails the caller.
pub(crate) fn hash_directory_entries(directory: &Path) -> Result<String, InferlabError> {
    let mut files = Vec::new();
    collect_regular_files(directory, directory, &mut files)?;
    // Byte-wise ordering over the `/`-joined relative path keeps the digest
    // stable across filesystems and platforms.
    files.sort_by(|left, right| left.0.cmp(&right.0));
    if files.is_empty() {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "image source directory {} enumerates no regular files",
                directory.display()
            ),
        });
    }
    let mut digest = Sha256::new();
    for (relative, path) in &files {
        let bytes = fs::read(path).map_err(|source| InferlabError::DatasetIo {
            operation: "read",
            path: path.clone(),
            source,
        })?;
        digest.update(relative);
        digest.update([0x00]);
        digest.update(Sha256::digest(&bytes));
        digest.update([0x00]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Collect each regular file under `current` as its `/`-joined
/// `root`-relative byte rendering paired with its full path.
fn collect_regular_files(
    root: &Path,
    current: &Path,
    files: &mut Vec<(Vec<u8>, std::path::PathBuf)>,
) -> Result<(), InferlabError> {
    let entries = fs::read_dir(current).map_err(|source| InferlabError::DatasetIo {
        operation: "enumerate",
        path: current.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| InferlabError::DatasetIo {
            operation: "enumerate",
            path: current.to_path_buf(),
            source,
        })?;
        let file_type = entry
            .file_type()
            .map_err(|source| InferlabError::DatasetIo {
                operation: "inspect",
                path: entry.path(),
                source,
            })?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_regular_files(root, &path, files)?;
        } else if file_type.is_file() {
            let relative =
                path.strip_prefix(root)
                    .map_err(|_| InferlabError::DatasetPreparation {
                        message: format!(
                            "image source entry {} escaped its directory {}",
                            path.display(),
                            root.display()
                        ),
                    })?;
            let mut rendered = Vec::new();
            for component in relative.components() {
                if !rendered.is_empty() {
                    rendered.push(b'/');
                }
                rendered.extend_from_slice(component.as_os_str().as_encoded_bytes());
            }
            files.push((rendered, path));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::hash_directory_entries;
    use sha2::{Digest, Sha256};
    use std::fs;

    fn reference_digest(root: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
        let mut files = Vec::new();
        fn collect(
            root: &std::path::Path,
            current: &std::path::Path,
            files: &mut Vec<String>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            for entry in fs::read_dir(current)? {
                let entry = entry?;
                // Match the production rule: entry file types do not follow
                // symbolic links.
                let file_type = entry.file_type()?;
                let path = entry.path();
                if file_type.is_dir() {
                    collect(root, &path, files)?;
                } else if file_type.is_file() {
                    files.push(
                        path.strip_prefix(root)
                            .map_err(|error| error.to_string())?
                            .to_string_lossy()
                            .replace('\\', "/"),
                    );
                }
            }
            Ok(())
        }
        collect(root, root, &mut files)?;
        files.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        let mut digest = Sha256::new();
        for relative in files {
            digest.update(relative.as_bytes());
            digest.update([0x00]);
            digest.update(Sha256::digest(fs::read(root.join(&relative))?));
            digest.update([0x00]);
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    #[test]
    fn directory_digest_is_deterministic_and_matches_the_documented_rule()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        fs::create_dir_all(directory.path().join("nested"))?;
        fs::write(directory.path().join("alpha.png"), b"alpha")?;
        fs::write(directory.path().join("nested/beta.png"), b"beta")?;

        let first = hash_directory_entries(directory.path())?;
        let second = hash_directory_entries(directory.path())?;
        assert_eq!(first, second);
        assert_eq!(first, reference_digest(directory.path())?);
        Ok(())
    }

    #[test]
    fn directory_digest_is_sensitive_to_content_and_names() -> Result<(), Box<dyn std::error::Error>>
    {
        let left = tempfile::tempdir()?;
        let right = tempfile::tempdir()?;
        fs::write(left.path().join("a.png"), b"alpha")?;
        fs::write(right.path().join("a.png"), b"alpha")?;
        assert_eq!(
            hash_directory_entries(left.path())?,
            hash_directory_entries(right.path())?
        );

        fs::write(right.path().join("a.png"), b"tampered")?;
        assert_ne!(
            hash_directory_entries(left.path())?,
            hash_directory_entries(right.path())?
        );

        fs::write(right.path().join("a.png"), b"alpha")?;
        fs::rename(right.path().join("a.png"), right.path().join("b.png"))?;
        assert_ne!(
            hash_directory_entries(left.path())?,
            hash_directory_entries(right.path())?
        );
        Ok(())
    }

    #[test]
    fn directory_digest_rejects_empty_and_missing_directories()
    -> Result<(), Box<dyn std::error::Error>> {
        let empty = tempfile::tempdir()?;
        let error = match hash_directory_entries(empty.path()) {
            Err(crate::InferlabError::DatasetPreparation { message }) => message,
            other => return Err(format!("expected DatasetPreparation, got {other:?}").into()),
        };
        assert!(error.contains("enumerates no regular files"), "{error}");

        let missing = empty.path().join("missing");
        match hash_directory_entries(&missing) {
            Err(crate::InferlabError::DatasetIo {
                operation, path, ..
            }) => {
                assert_eq!(operation, "enumerate");
                assert_eq!(path, missing);
            }
            other => return Err(format!("expected DatasetIo, got {other:?}").into()),
        }
        Ok(())
    }

    // Symbolic links are not followed: a linked file contributes nothing, and
    // a directory holding only links enumerates no regular files
    // ([[RFC-0004:C-BENCH-REQUEST-SOURCES]]).
    #[cfg(unix)]
    #[test]
    fn directory_digest_skips_symbolic_links() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        fs::write(directory.path().join("real.png"), b"real")?;
        std::os::unix::fs::symlink(
            directory.path().join("real.png"),
            directory.path().join("linked.png"),
        )?;
        assert_eq!(
            hash_directory_entries(directory.path())?,
            reference_digest(directory.path())?
        );

        let plain = tempfile::tempdir()?;
        fs::write(plain.path().join("real.png"), b"real")?;
        assert_eq!(
            hash_directory_entries(directory.path())?,
            hash_directory_entries(plain.path())?,
            "a symlink must not change the digest"
        );

        let links_only = tempfile::tempdir()?;
        std::os::unix::fs::symlink(
            directory.path().join("real.png"),
            links_only.path().join("linked.png"),
        )?;
        match hash_directory_entries(links_only.path()) {
            Err(crate::InferlabError::DatasetPreparation { message }) => {
                assert!(message.contains("enumerates no regular files"), "{message}");
            }
            other => return Err(format!("expected DatasetPreparation, got {other:?}").into()),
        }
        Ok(())
    }
}
