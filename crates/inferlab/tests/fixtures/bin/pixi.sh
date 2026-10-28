#!/bin/sh
# Fake `pixi` for every test harness, answering the invocations InferLab
# issues as pixi 0.81 does: `info --json`, the toolchain `install`,
# `list --json`, and the `run` forms that activate a named environment.
# Switches (all optional):
#   FAKE_PIXI_LOG             append every argument vector
#   FAKE_PIXI_INSTALL_LOG     append each toolchain install's arguments
#   PIXI_FIXTURE_PLATFORM, PIXI_FIXTURE_GLIBC   shape `info --json`
#   PIXI_FIXTURE_LIST         a file answering `list --json`
#   FAKE_PIXI_PROBE_EXIT      exit status of the `--locked --no-install` usability probe
#   FAKE_PIXI_UNAVAILABLE=1   the probe reports a missing environment prefix
#   FIXTURE_PIXI_BUILD=mutate|fail   a wheel build leaves a stray file in the
#                             source tree, and with fail also crashes
#   FIXTURE_PATH_MAPS, FIXTURE_WHEEL_EXTRA      observe and shape wheel builds
if [ -n "${FAKE_PIXI_LOG:-}" ]; then
  printf '%s\n' "$*" >> "$FAKE_PIXI_LOG"
fi
if [ "$1" = info ] && [ "$2" = --json ]; then
  case "$(uname -m)" in
    x86_64) detected_platform=linux-64 ;;
    aarch64) detected_platform=linux-aarch64 ;;
    *) detected_platform=unsupported ;;
  esac
  if [ "${PIXI_FIXTURE_GLIBC:-1}" = 1 ]; then
    virtual_packages='["__unix=0=0","__linux=6.11.0=0","__glibc=2.35=0"]'
  else
    virtual_packages='["__unix=0=0","__linux=6.11.0=0"]'
  fi
  printf '{"platform":"%s","virtual_packages":%s}\n' \
    "${PIXI_FIXTURE_PLATFORM:-$detected_platform}" "$virtual_packages"
  exit 0
fi
if [ "$1" = install ] && [ "$2" = --manifest-path ] && [ "$4" = --all ] && [ "$5" = --locked ]; then
  if [ -n "${FAKE_PIXI_INSTALL_LOG:-}" ]; then
    printf '%s\n' "$*" >> "$FAKE_PIXI_INSTALL_LOG"
  fi
  prefix="$(dirname "$3")"
  mkdir -p "$prefix/.pixi/envs/eval/bin" "$prefix/.pixi/envs/bench/bin"
  cat > "$prefix/.pixi/envs/eval/bin/python" <<'PYTHON'
#!/bin/sh
if [ "$2" = --handshake ]; then
  printf '{"lm_eval_version":"0.4.13"}\n'
  exit 0
fi
shift
exec fixture-eval-client "$@"
PYTHON
  cat > "$prefix/.pixi/envs/bench/bin/python" <<'PYTHON'
#!/bin/sh
if [ "$2" = --handshake ]; then
  printf '{"aiperf_version":"0.13.0+inferlab.3","transformers_version":"5.12.1"}\n'
  exit 0
fi
if [ "$1" = -m ] && [ "$2" = inferlab_bench_runner.bench_client ]; then
  shift 2
else
  shift
fi
exec fixture-bench-client "$@"
PYTHON
  chmod +x "$prefix/.pixi/envs/eval/bin/python" "$prefix/.pixi/envs/bench/bin/python"
  # The image-packaging tools print what the release-pinned tools print;
  # `wheel unpack` keeps the real `<name>-<version>/` layout over the
  # fixture's text payloads, which carry no ELF members.
  mkdir -p "$prefix/.pixi/envs/image/bin"
  cat > "$prefix/.pixi/envs/image/bin/python" <<'PYTHON'
#!/bin/sh
if [ "$1" = -m ] && [ "$2" = wheel ]; then
  case "$3" in
    version) printf 'wheel 0.48.0\n'; exit 0 ;;
    unpack)
      dir="$6/$(basename "$4" .whl | cut -d- -f1-2)"
      mkdir -p "$dir" && cp "$4" "$dir/payload.txt"
      printf 'Unpacking to: %s...OK\n' "$dir"
      exit 0 ;;
    pack)
      cp "$4/payload.txt" "$6/$(basename "$4")-py3-none-any.whl"
      exit 0 ;;
  esac
fi
printf 'unexpected image python fixture arguments: %s\n' "$*" >&2
exit 2
PYTHON
  printf '%s\n' '#!/bin/sh' 'printf "patchelf 0.19.2\n"' > "$prefix/.pixi/envs/image/bin/patchelf"
  printf '%s\n' '#!/bin/sh' 'printf "cuobjdump: NVIDIA (R) fat binary listing tool\nCopyright (c) 2005-2026 NVIDIA Corporation\nBuilt on Tue_Sep_01_08:45:21_PDT_2026\nCuda compilation tools, release 13.4, V13.4.92\nBuild cuda_13.4.r13.4/compiler.38855100_0\n"' > "$prefix/.pixi/envs/image/bin/cuobjdump"
  chmod +x "$prefix/.pixi/envs/image/bin/python" "$prefix/.pixi/envs/image/bin/patchelf" "$prefix/.pixi/envs/image/bin/cuobjdump"
  exit 0
fi
if [ "$1" = list ] && [ "$2" = --json ]; then
  if [ -n "${PIXI_FIXTURE_LIST:-}" ]; then
    cat "$PIXI_FIXTURE_LIST"
    exit 0
  fi
  # Mirrors the pixi 0.81 row shape: source-backed projects carry no hash
  # and a workspace-relative url; there is no editable field.
  cat <<'JSON'
[
  {"name": "python", "kind": "conda", "url": "https://conda.example/linux-64/python-3.12.0.conda", "sha256": "1111111111111111111111111111111111111111111111111111111111111111", "source": "https://conda.example"},
  {"name": "inferlab-integration-vllm", "kind": "pypi", "url": "https://pypi.example/inferlab_integration_vllm-0.1.0-py3-none-any.whl", "sha256": "2222222222222222222222222222222222222222222222222222222222222222", "source": "https://pypi.example/simple"},
  {"name": "vllm", "kind": "pypi", "url": "./vendor/vllm", "sha256": null, "source": "./vendor/vllm"},
  {"name": "flashinfer", "kind": "pypi", "url": "./vendor/flashinfer", "sha256": null, "source": "./vendor/flashinfer"}
]
JSON
  exit 0
fi
if [ "$1" = run ] && [ "$2" = --locked ] && [ "$3" = --no-install ] && [ "$4" = --executable ] && [ "$5" = -e ] && [ "$7" = -- ]; then
  shift 7
  if [ -n "${FAKE_PIXI_PROBE_EXIT:-}" ]; then
    exit "$FAKE_PIXI_PROBE_EXIT"
  fi
  if [ "${FAKE_PIXI_UNAVAILABLE:-0}" = 1 ] && [ "$1" = true ]; then
    printf 'environment prefix is missing\n' >&2
    exit 3
  fi
elif [ "$1" = run ] && [ "$2" = --clean-env ] && [ "$3" = --as-is ] && [ "$4" = --executable ] && [ "$5" = -e ] && [ "$7" = -- ]; then
  shift 7
elif [ "$1" = run ] && [ "$2" = --as-is ] && [ "$3" = --executable ] && [ "$4" = -e ] && [ "$6" = -- ]; then
  shift 6
elif [ "$1" = -q ] && [ "$2" = run ] && [ "$3" = --as-is ] && [ "$4" = --executable ] && [ "$5" = --manifest-path ] && [ "$7" = -e ] && [ "$9" = -- ]; then
  shift 9
else
  printf 'unexpected pixi fixture arguments: %s\n' "$*" >&2
  exit 2
fi
if [ "$1" = /bin/sh ] && [ "$2" = -c ]; then
  shift 4
  while [ $# -gt 0 ] && printf '%s' "$1" | grep -q =; do shift; done
fi
# The second stage appends the compiler path maps:
# `sh -c SCRIPT sh HOST NVCC RUST RUST_ENCODED`.
if [ "$1" = /bin/sh ] && [ "$2" = -c ]; then
  printf '%s\n' "$5" > "${FIXTURE_PATH_MAPS:-/dev/null}"
  shift 8
fi
# The third stage captures the build environment: `sh -c SCRIPT sh FILE`.
if [ "$1" = /bin/sh ] && [ "$2" = -c ]; then
  env > "$5"
  shift 5
fi
if [ "$1" = python ] && [ "$3" = pip ] && [ "$4" = wheel ] && [ "$7" = --wheel-dir ]; then
  case "${FIXTURE_PIXI_BUILD:-}" in
    mutate|fail) printf 'stray\n' > vendor/vllm/stray-build-artifact.txt ;;
  esac
  if [ "${FIXTURE_PIXI_BUILD:-}" = fail ]; then
    printf 'simulated backend crash\n' >&2
    exit 1
  fi
  # Like pip, name the wheel after the project metadata when the build
  # directory declares it; bare fixture trees fall back to the directory name.
  name="$(sed -n 's/^name = "\(.*\)"$/\1/p' "$9/pyproject.toml" 2>/dev/null | head -n 1)"
  [ -n "$name" ] || name="$(basename "$9")"
  printf 'wheel bytes for %s\n' "$name" > "$8/${name}-1.0-py3-none-any.whl"
  # The payload lists the build tree, so tests can see what the build read.
  (cd "$9" && find . -mindepth 1 -not -path './.git*' | sort) >> "$8/${name}-1.0-py3-none-any.whl"
  # Lets a test plant content a real build would embed.
  if [ -n "${FIXTURE_WHEEL_EXTRA:-}" ]; then
    printf '%s\n' "$FIXTURE_WHEEL_EXTRA" >> "$8/${name}-1.0-py3-none-any.whl"
  fi
  exit 0
fi
exec "$@"
