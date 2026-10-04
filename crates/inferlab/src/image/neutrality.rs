//! Path-neutral image packages ([[RFC-0007:C-IMAGE-BUILD]], [[ADR-0057]]):
//! compiler path maps for package builds, library search paths rewritten
//! after the build, and verification of every package before it is cached
//! or assembled. The tools come from the toolchain's image-packaging runtime.

use super::entrypoint::ENV_PREFIX;
use crate::InferlabError;
use crate::toolchain::{ImageToolchainIdentity, InstalledImageToolchain};
use object::read::elf::ElfFile64;
use object::{Object, ObjectSection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Where the sanitized source tree appears in built packages.
pub(super) const SOURCE_TARGET: &str = "/opt/inferlab-src";
/// Where any other workspace path appears in built packages.
pub(super) const WORKSPACE_TARGET: &str = "/opt/inferlab-workspace";

/// One compiler path map.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PathMap {
    pub from: String,
    pub to: String,
}

/// One rewritten or removed library search path list of an ELF member.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SearchPathRewrite {
    pub member: String,
    /// `rpath` or `runpath`, kept as found.
    pub kind: String,
    pub before: Vec<String>,
    pub after: Vec<String>,
}

/// What made one package path-neutral and what verified it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PackageNeutrality {
    pub runtime: ImageToolchainIdentity,
    pub path_maps: Vec<PathMap>,
    /// Rewrites this build performed; a reused package was rewritten by the
    /// build that produced it and is only verified here.
    pub search_paths: Vec<SearchPathRewrite>,
    pub verified_members: u64,
    pub decoded_sections: u64,
    pub device_code_images: u64,
}

/// The paths a package build must not leak, and their mapped targets.
pub(super) struct BuildPaths {
    pub workspace_root: PathBuf,
    pub env_prefix: PathBuf,
    pub source_root: PathBuf,
    pub build_dir: PathBuf,
    pub home: Option<PathBuf>,
}

/// A path and its canonical form when they differ, so a symlinked view and
/// its target are both covered.
fn spellings(path: &Path) -> Vec<String> {
    let mut forms = vec![path.display().to_string()];
    if let Ok(canonical) = path.canonicalize() {
        let canonical = canonical.display().to_string();
        if !forms.contains(&canonical) {
            forms.push(canonical);
        }
    }
    forms
}

impl BuildPaths {
    /// Compiler path maps, fallback first: the compilers apply the last
    /// matching map, so the most specific ones come last.
    pub(super) fn path_maps(&self) -> Result<Vec<PathMap>, InferlabError> {
        let mut maps = Vec::new();
        for (path, target) in [
            (&self.workspace_root, WORKSPACE_TARGET),
            (&self.env_prefix, ENV_PREFIX),
            (&self.source_root, SOURCE_TARGET),
        ] {
            for from in spellings(path) {
                // The maps travel through whitespace-split flag variables and
                // nvcc's comma-split -Xcompiler, and `=` delimits a map.
                if from.chars().any(|character| {
                    character.is_whitespace()
                        || matches!(character, ',' | '=' | '"' | '\'' | '$' | '`' | '\\')
                }) {
                    return Err(InferlabError::ImageBuild {
                        message: format!(
                            "package-build path {from:?} contains a character that compiler \
                             path maps cannot carry (whitespace, comma, equals sign, quote, \
                             dollar, backquote, or backslash); move the workspace to a plainer path"
                        ),
                    });
                }
                maps.push(PathMap {
                    from,
                    to: target.to_owned(),
                });
            }
        }
        Ok(maps)
    }

    fn needles(&self) -> Vec<(&'static str, String)> {
        let mut needles = Vec::new();
        for (label, path) in [
            ("package-build directory", &self.build_dir),
            ("Pixi environment prefix", &self.env_prefix),
            ("workspace root", &self.workspace_root),
        ] {
            for form in spellings(path) {
                needles.push((label, form));
            }
        }
        // A home directory matches only at a path-component boundary, and a
        // degenerate one (`/`) would match everything.
        if let Some(home) = &self.home
            && home.components().count() > 1
        {
            for form in spellings(home) {
                needles.push(("home directory", format!("{}/", form.trim_end_matches('/'))));
            }
        }
        needles
    }
}

/// `-ffile-prefix-map` options for the host compilers, in map order.
pub(super) fn host_flags(maps: &[PathMap]) -> String {
    maps.iter()
        .map(|map| format!("-ffile-prefix-map={}={}", map.from, map.to))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The same maps for nvcc, which hands them to its host compiler.
pub(super) fn nvcc_flags(maps: &[PathMap]) -> String {
    maps.iter()
        .map(|map| format!("-Xcompiler -ffile-prefix-map={}={}", map.from, map.to))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The rewritten search path list: entries under the environment prefix
/// move to the image environment prefix, entries naming a leaked path are
/// removed, and every other entry (such as `$ORIGIN`) is kept.
pub(super) fn rewrite_search_paths(entries: &[String], paths: &BuildPaths) -> Vec<String> {
    let prefixes = spellings(&paths.env_prefix);
    let needles = paths.needles();
    entries
        .iter()
        .filter_map(|entry| {
            for prefix in &prefixes {
                if entry == prefix {
                    return Some(ENV_PREFIX.to_owned());
                }
                if let Some(rest) = entry.strip_prefix(prefix.as_str())
                    && rest.starts_with('/')
                {
                    return Some(format!("{ENV_PREFIX}{rest}"));
                }
            }
            if needles
                .iter()
                .any(|(_, needle)| entry.contains(needle.trim_end_matches('/')))
            {
                return None;
            }
            Some(entry.clone())
        })
        .collect()
}

/// The first leaked path in `bytes`, by label.
fn leak(needles: &[(&'static str, String)], bytes: &[u8]) -> Option<&'static str> {
    needles
        .iter()
        .find(|(_, needle)| memchr::memmem::find(bytes, needle.as_bytes()).is_some())
        .map(|(label, _)| *label)
}

/// Unpack, rewrite search paths when `rewrite` is set, repack, and verify
/// one wheel in place. `work` is scratch space the caller owns.
pub(super) fn make_neutral(
    wheel: &Path,
    tools: &InstalledImageToolchain,
    paths: &BuildPaths,
    work: &Path,
    rewrite: bool,
) -> Result<PackageNeutrality, InferlabError> {
    let path_maps = paths.path_maps()?;
    let needles = paths.needles();
    if work.exists() {
        std::fs::remove_dir_all(work).map_err(|source| io("clear", work, source))?;
    }
    let unpacked = work.join("unpacked");
    std::fs::create_dir_all(&unpacked).map_err(|source| io("create", &unpacked, source))?;
    run(
        Command::new(&tools.python)
            .args(["-m", "wheel", "unpack"])
            .arg(wheel)
            .arg("-d")
            .arg(&unpacked),
        "unpack a built package",
    )?;
    let root = single_directory(&unpacked)?;
    let members = files(&root)?;

    let mut search_paths = Vec::new();
    if rewrite {
        for member in &members {
            let Some(rewrite) = rewrite_member(member, &root, tools, paths)? else {
                continue;
            };
            search_paths.push(rewrite);
        }
        if !search_paths.is_empty() {
            let packed = work.join("packed");
            std::fs::create_dir_all(&packed).map_err(|source| io("create", &packed, source))?;
            run(
                Command::new(&tools.python)
                    .args(["-m", "wheel", "pack"])
                    .arg(&root)
                    .arg("-d")
                    .arg(&packed),
                "repack a built package",
            )?;
            let repacked = files(&packed)?;
            let [repacked] = repacked.as_slice() else {
                return Err(InferlabError::ImageBuild {
                    message: format!(
                        "repacking {} produced {} files",
                        wheel.display(),
                        repacked.len()
                    ),
                });
            };
            if repacked.file_name() != wheel.file_name() {
                return Err(InferlabError::ImageBuild {
                    message: format!(
                        "repacking {} produced {}, a different name",
                        wheel.display(),
                        repacked.display()
                    ),
                });
            }
            std::fs::rename(repacked, wheel).map_err(|source| io("replace", wheel, source))?;
        }
    }

    let mut verified_members = 0;
    let mut decoded_sections = 0;
    let mut device_code_images = 0;
    let device = work.join("device");
    for member in &members {
        let name = relative(member, &root);
        let bytes = std::fs::read(member).map_err(|source| io("read", member, source))?;
        verified_members += 1;
        if let Some(label) = leak(&needles, &bytes) {
            return Err(leaked(wheel, &name, label));
        }
        if !bytes.starts_with(b"\x7fELF") {
            continue;
        }
        decoded_sections += verify_compressed_sections(&bytes, &needles, wheel, &name)?;
        if has_fatbin(&bytes, wheel, &name)? {
            let images = extract_device_code(member, tools, &device)?;
            if images.is_empty() {
                return Err(InferlabError::ImageBuild {
                    message: format!(
                        "built package {} member {name} carries a fatbin from which no device \
                         code could be extracted for verification",
                        file_label(wheel)
                    ),
                });
            }
            for image in images {
                let content = std::fs::read(&image).map_err(|source| io("read", &image, source))?;
                device_code_images += 1;
                if let Some(label) = leak(&needles, &content) {
                    return Err(leaked(wheel, &format!("{name} (device code)"), label));
                }
                if content.starts_with(b"\x7fELF") {
                    decoded_sections +=
                        verify_compressed_sections(&content, &needles, wheel, &name)?;
                }
            }
            std::fs::remove_dir_all(&device).map_err(|source| io("remove", &device, source))?;
        }
    }
    std::fs::remove_dir_all(work).map_err(|source| io("remove", work, source))?;
    Ok(PackageNeutrality {
        runtime: tools.identity.clone(),
        path_maps,
        search_paths,
        verified_members,
        decoded_sections,
        device_code_images,
    })
}

/// Rewrite one ELF member's search paths with patchelf, keeping their kind.
fn rewrite_member(
    member: &Path,
    root: &Path,
    tools: &InstalledImageToolchain,
    paths: &BuildPaths,
) -> Result<Option<SearchPathRewrite>, InferlabError> {
    let bytes = std::fs::read(member).map_err(|source| io("read", member, source))?;
    if !bytes.starts_with(b"\x7fELF") {
        return Ok(None);
    }
    let Some((kind, before)) =
        search_paths(&bytes).map_err(|message| InferlabError::ImageBuild {
            message: format!("{}: {message}", member.display()),
        })?
    else {
        return Ok(None);
    };
    let after = rewrite_search_paths(&before, paths);
    if after == before {
        return Ok(None);
    }
    let mut command = Command::new(&tools.patchelf);
    if after.is_empty() {
        command.arg("--remove-rpath");
    } else {
        if kind == "rpath" {
            command.arg("--force-rpath");
        }
        command.arg("--set-rpath").arg(after.join(":"));
    }
    run(command.arg(member), "rewrite library search paths")?;
    Ok(Some(SearchPathRewrite {
        member: relative(member, root),
        kind: kind.to_owned(),
        before,
        after,
    }))
}

/// The search path kind and entries of a 64-bit ELF file, if it has any.
fn search_paths(bytes: &[u8]) -> Result<Option<(&'static str, Vec<String>)>, String> {
    let Ok(file) = ElfFile64::<object::Endianness>::parse(bytes) else {
        // Not a 64-bit ELF file: nothing to rewrite, still verified as stored.
        return Ok(None);
    };
    let endian = file.endian();
    let table = file
        .elf_section_table()
        .dynamic_table(endian, file.data())
        .map_err(|error| format!("unreadable dynamic section: {error}"))?;
    for entry in &table {
        let kind = match entry.tag {
            tag if tag == object::elf::DT_RPATH => "rpath",
            tag if tag == object::elf::DT_RUNPATH => "runpath",
            _ => continue,
        };
        let value = table
            .string(entry)
            .map_err(|error| format!("unreadable search path: {error}"))?;
        let entries = String::from_utf8_lossy(value)
            .split(':')
            .filter(|entry| !entry.is_empty())
            .map(str::to_owned)
            .collect();
        return Ok(Some((kind, entries)));
    }
    Ok(None)
}

/// Decode every compressed section and scan it; one that cannot be decoded
/// fails the package.
fn verify_compressed_sections(
    bytes: &[u8],
    needles: &[(&'static str, String)],
    wheel: &Path,
    member: &str,
) -> Result<u64, InferlabError> {
    let file = object::File::parse(bytes).map_err(|error| InferlabError::ImageBuild {
        message: format!(
            "built package {} member {member} is an ELF file that cannot be read: {error}",
            file_label(wheel)
        ),
    })?;
    let mut decoded = 0;
    for section in file.sections() {
        let range = section
            .compressed_file_range()
            .map_err(|error| undecodable(wheel, member, &section, error))?;
        if range.format == object::CompressionFormat::None {
            continue;
        }
        let data = section
            .uncompressed_data()
            .map_err(|error| undecodable(wheel, member, &section, error))?;
        decoded += 1;
        if let Some(label) = leak(needles, &data) {
            let name = section.name().unwrap_or("?");
            return Err(leaked(wheel, &format!("{member} ({name})"), label));
        }
    }
    Ok(decoded)
}

fn has_fatbin(bytes: &[u8], wheel: &Path, member: &str) -> Result<bool, InferlabError> {
    let file = object::File::parse(bytes).map_err(|error| InferlabError::ImageBuild {
        message: format!(
            "built package {} member {member} is an ELF file that cannot be read: {error}",
            file_label(wheel)
        ),
    })?;
    Ok(file.section_by_name(".nv_fatbin").is_some())
}

/// Extract every device code image of one ELF file into `into`.
fn extract_device_code(
    member: &Path,
    tools: &InstalledImageToolchain,
    into: &Path,
) -> Result<Vec<PathBuf>, InferlabError> {
    if into.exists() {
        std::fs::remove_dir_all(into).map_err(|source| io("clear", into, source))?;
    }
    std::fs::create_dir_all(into).map_err(|source| io("create", into, source))?;
    run(
        Command::new(&tools.cuobjdump)
            .args(["-xelf", "all", "-xptx", "all"])
            .arg(member)
            .current_dir(into),
        "extract device code",
    )?;
    files(into)
}

fn run(command: &mut Command, operation: &str) -> Result<(), InferlabError> {
    let output = command
        .output()
        .map_err(|source| InferlabError::ImageToolLaunch {
            operation: operation.to_owned(),
            source,
        })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(InferlabError::ImageToolExit {
            operation: operation.to_owned(),
            status: output.status,
            diagnostics: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        })
    }
}

/// The one directory `wheel unpack` creates, `<name>-<version>`.
fn single_directory(parent: &Path) -> Result<PathBuf, InferlabError> {
    let entries = std::fs::read_dir(parent)
        .map_err(|source| io("read", parent, source))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    match entries.as_slice() {
        [single] if single.is_dir() => Ok(single.clone()),
        _ => Err(InferlabError::ImageBuild {
            message: format!(
                "unpacking a built package left {} entries in {}, expected one directory",
                entries.len(),
                parent.display()
            ),
        }),
    }
}

/// Every regular file below `root`, in path order.
fn files(root: &Path) -> Result<Vec<PathBuf>, InferlabError> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in
            std::fs::read_dir(&directory).map_err(|source| io("read", &directory, source))?
        {
            let entry = entry.map_err(|source| io("read", &directory, source))?;
            let kind = entry
                .file_type()
                .map_err(|source| io("inspect", &entry.path(), source))?;
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file() {
                found.push(entry.path());
            }
        }
    }
    found.sort();
    Ok(found)
}

fn relative(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn file_label(wheel: &Path) -> String {
    wheel.file_name().map_or_else(
        || wheel.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

fn leaked(wheel: &Path, member: &str, label: &str) -> InferlabError {
    InferlabError::ImageBuild {
        message: format!(
            "built package {} member {member} embeds the {label}; images are published, so \
             every built path must map to a fixed path ([[RFC-0007:C-IMAGE-BUILD]])",
            file_label(wheel)
        ),
    }
}

fn undecodable(
    wheel: &Path,
    member: &str,
    section: &object::Section<'_, '_>,
    error: object::read::Error,
) -> InferlabError {
    InferlabError::ImageBuild {
        message: format!(
            "built package {} member {member} section {} cannot be decoded for verification: {error}",
            file_label(wheel),
            section.name().unwrap_or("?")
        ),
    }
}

fn io(operation: &'static str, path: &Path, source: std::io::Error) -> InferlabError {
    InferlabError::EnvironmentIo {
        path: path.to_path_buf(),
        operation,
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BuildPaths, PathMap, host_flags, leak, nvcc_flags, rewrite_search_paths,
        verify_compressed_sections,
    };
    use std::io::Write;
    use std::path::{Path, PathBuf};

    /// An ELF object whose `.debug_info` is a zlib-compressed section
    /// (`SHF_COMPRESSED`) holding `stream` as its compressed payload.
    fn elf_with_compressed_debug(stream: &[u8], size: u64) -> Result<Vec<u8>, String> {
        use object::write::Object;
        let mut file = Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        let section = file.add_section(
            Vec::new(),
            b".debug_info".to_vec(),
            object::SectionKind::Debug,
        );
        file.section_mut(section).flags = object::SectionFlags::Elf {
            sh_type: object::elf::SHT_PROGBITS,
            sh_flags: object::elf::SHF_COMPRESSED,
        };
        // Elf64_Chdr: ch_type, ch_reserved, ch_size, ch_addralign.
        let mut data = Vec::new();
        data.extend_from_slice(&object::elf::ELFCOMPRESS_ZLIB.0.to_le_bytes());
        data.extend_from_slice(&0_u32.to_le_bytes());
        data.extend_from_slice(&size.to_le_bytes());
        data.extend_from_slice(&1_u64.to_le_bytes());
        data.extend_from_slice(stream);
        file.append_section_data(section, &data, 1);
        file.write().map_err(|error| error.to_string())
    }

    fn zlib(bytes: &[u8]) -> Result<Vec<u8>, String> {
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder
            .write_all(bytes)
            .map_err(|error| error.to_string())?;
        encoder.finish().map_err(|error| error.to_string())
    }

    #[test]
    fn compressed_debug_sections_are_decoded_before_they_are_verified() -> Result<(), String> {
        let needles = paths().needles();
        let wheel = Path::new("demo-1.0-py3-none-any.whl");
        let content = b"DW_AT_comp_dir /srv/example/ws/.inferlab/records/r/build/sources/demo";
        let leaking = elf_with_compressed_debug(&zlib(content)?, content.len() as u64)?;
        assert!(
            leak(&needles, &leaking).is_none(),
            "compression hides the path from a plain scan"
        );
        let error = verify_compressed_sections(&leaking, &needles, wheel, "demo/_C.so")
            .err()
            .map(|error| error.to_string())
            .ok_or("a leaked compressed section must fail")?;
        assert!(
            error.contains("demo/_C.so (.debug_info)") && error.contains("package-build directory"),
            "{error}"
        );

        let clean = b"DW_AT_comp_dir /opt/inferlab-src/demo";
        let decoded = verify_compressed_sections(
            &elf_with_compressed_debug(&zlib(clean)?, clean.len() as u64)?,
            &needles,
            wheel,
            "demo/_C.so",
        )
        .map_err(|error| error.to_string())?;
        assert_eq!(decoded, 1);

        let garbled = elf_with_compressed_debug(b"not a zlib stream", 64)?;
        assert!(
            verify_compressed_sections(&garbled, &needles, wheel, "demo/_C.so").is_err(),
            "a section that cannot be decoded fails the package"
        );
        Ok(())
    }

    fn paths() -> BuildPaths {
        BuildPaths {
            workspace_root: PathBuf::from("/srv/example/ws"),
            env_prefix: PathBuf::from("/srv/example/ws/.pixi/envs/serve"),
            source_root: PathBuf::from("/srv/example/ws/.inferlab/records/r/build/sources"),
            build_dir: PathBuf::from("/srv/example/ws/.inferlab/records/r/build"),
            home: Some(PathBuf::from("/srv/example")),
        }
    }

    #[test]
    fn the_most_specific_map_comes_last_because_compilers_apply_the_last_match()
    -> Result<(), String> {
        let maps = paths().path_maps().map_err(|error| error.to_string())?;
        let targets = maps.iter().map(|map| map.to.as_str()).collect::<Vec<_>>();
        assert_eq!(
            targets,
            [
                "/opt/inferlab-workspace",
                "/opt/inferlab-env",
                "/opt/inferlab-src"
            ]
        );
        let one = [PathMap {
            from: "/a".to_owned(),
            to: "/b".to_owned(),
        }];
        assert_eq!(host_flags(&one), "-ffile-prefix-map=/a=/b");
        assert_eq!(nvcc_flags(&one), "-Xcompiler -ffile-prefix-map=/a=/b");

        let mut odd = paths();
        odd.workspace_root = PathBuf::from("/srv/my ws");
        assert!(odd.path_maps().is_err(), "a space would split the flag");
        Ok(())
    }

    #[test]
    fn search_paths_move_to_the_image_environment_and_leaks_are_dropped() {
        let entries = [
            "/srv/example/ws/.pixi/envs/serve/lib".to_owned(),
            "$ORIGIN/../lib".to_owned(),
            "/srv/example/ws/.inferlab/records/r/build/sources/vllm/build".to_owned(),
            "/usr/lib/x86_64-linux-gnu".to_owned(),
        ];
        assert_eq!(
            rewrite_search_paths(&entries, &paths()),
            [
                "/opt/inferlab-env/lib",
                "$ORIGIN/../lib",
                "/usr/lib/x86_64-linux-gnu"
            ]
        );
    }

    #[test]
    fn a_home_directory_matches_only_at_a_component_boundary() {
        let needles = paths().needles();
        assert_eq!(
            leak(&needles, b"tail /srv/example/other/file"),
            Some("home directory")
        );
        assert_eq!(leak(&needles, b"/srv/examples/file"), None);
        assert_eq!(
            leak(&needles, b"/srv/example/ws/.pixi/envs/serve/include/x.h"),
            Some("Pixi environment prefix")
        );
    }
}
