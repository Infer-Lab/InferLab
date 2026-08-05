from __future__ import annotations

import hashlib
import pathlib
import sys
import tempfile
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from product_release_assets import (  # noqa: E402
    PackageIdentity,
    ReleaseAssetError,
    prepare_assets,
    verify_aggregate_assets,
    verify_repository_assets,
    verify_wheel_assets,
)


class ProductReleaseAssetTests(unittest.TestCase):
    @staticmethod
    def _write_repository_assets(directory: pathlib.Path) -> None:
        for filename in (
            "inferlab-x86_64-linux",
            "inferlab-aarch64-linux",
            "install.sh",
            "inferlab-plugin.tar.gz",
        ):
            content = f"qualified {filename}".encode()
            (directory / filename).write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            (directory / f"{filename}.sha256").write_text(f"{digest}  {filename}\n")
        (directory / "LICENSE").write_text("license")

    def test_prepares_the_complete_wheel_and_checksum_inventory(self) -> None:
        packages = [
            PackageIdentity("inferlab-adapter-sdk", "0.6.1"),
            PackageIdentity("inferlab-integration-vllm", "0.5.2"),
        ]
        wheels = {
            package.name: f"{package.name.replace('-', '_')}-{package.version}-py3-none-any.whl"
            for package in packages
        }
        contents = {
            filename: f"canonical bytes for {filename}".encode() for filename in wheels.values()
        }

        def read_json(url: str) -> object:
            package = next(package for package in packages if f"/{package.name}/" in url)
            filename = wheels[package.name]
            return {
                "info": {"name": package.name, "version": package.version},
                "urls": [
                    {
                        "filename": filename,
                        "packagetype": "bdist_wheel",
                        "url": f"https://files.example/{filename}",
                        "yanked": False,
                        "digests": {"sha256": hashlib.sha256(contents[filename]).hexdigest()},
                    },
                    {"filename": f"{package.name}.tar.gz", "packagetype": "sdist"},
                ],
            }

        def read_bytes(url: str) -> bytes:
            return contents[url.rsplit("/", 1)[1]]

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "assets"
            candidates = root / "candidates"
            candidates.mkdir()
            prepared = prepare_assets(packages, output, candidates, read_json, read_bytes)

            expected_names = {
                name for filename in wheels.values() for name in (filename, f"{filename}.sha256")
            }
            self.assertEqual({path.name for path in prepared}, expected_names)
            self.assertEqual({path.name for path in output.iterdir()}, expected_names)
            for filename, content in contents.items():
                digest = hashlib.sha256(content).hexdigest()
                self.assertEqual((output / filename).read_bytes(), content)
                self.assertEqual(
                    (output / f"{filename}.sha256").read_text(),
                    f"{digest}  {filename}\n",
                )

    def test_rejects_an_incomplete_or_noncanonical_index_before_publication(self) -> None:
        package = PackageIdentity("inferlab-adapter-sdk", "0.6.1")
        filename = "inferlab_adapter_sdk-0.6.1-py3-none-any.whl"
        content = b"canonical wheel"
        digest = hashlib.sha256(content).hexdigest()
        valid_wheel = {
            "filename": filename,
            "packagetype": "bdist_wheel",
            "url": f"https://files.example/{filename}",
            "yanked": False,
            "digests": {"sha256": digest},
        }
        cases = {
            "missing": [],
            "multiple": [valid_wheel, valid_wheel],
            "digest-mismatch": [
                {
                    **valid_wheel,
                    "digests": {"sha256": "0" * 64},
                }
            ],
        }

        for name, urls in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                output = root / "assets"
                candidates = root / "candidates"
                candidates.mkdir()
                payload = {
                    "info": {"name": package.name, "version": package.version},
                    "urls": urls,
                }

                def read_case_json(_url: str, payload: object = payload) -> object:
                    return payload

                with self.assertRaises(ReleaseAssetError):
                    prepare_assets(
                        [package],
                        output,
                        candidates,
                        read_case_json,
                        lambda _url: content,
                    )
                self.assertFalse(output.exists())

    def test_local_candidate_precedes_index_fallback_in_one_closed_inventory(self) -> None:
        candidate = PackageIdentity("inferlab-adapter-sdk", "0.6.2")
        published = PackageIdentity("inferlab-integration-vllm", "0.5.2")
        candidate_name = "inferlab_adapter_sdk-0.6.2-py3-none-any.whl"
        published_name = "inferlab_integration_vllm-0.5.2-py3-none-any.whl"
        candidate_content = b"qualified local candidate"
        published_content = b"canonical index wheel"

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidates = root / "candidates"
            output = root / "assets"
            candidates.mkdir()
            (candidates / candidate_name).write_bytes(candidate_content)
            candidate_digest = hashlib.sha256(candidate_content).hexdigest()
            (candidates / f"{candidate_name}.sha256").write_text(
                f"{candidate_digest}  {candidate_name}\n"
            )

            def read_json(url: str) -> object:
                self.assertIn(published.name, url)
                return {
                    "info": {"name": published.name, "version": published.version},
                    "urls": [
                        {
                            "filename": published_name,
                            "packagetype": "bdist_wheel",
                            "url": f"https://files.example/{published_name}",
                            "yanked": False,
                            "digests": {"sha256": hashlib.sha256(published_content).hexdigest()},
                        }
                    ],
                }

            prepared = prepare_assets(
                [candidate, published],
                output,
                candidates,
                read_json,
                lambda _url: published_content,
            )

            self.assertEqual(len(prepared), 4)
            self.assertEqual((output / candidate_name).read_bytes(), candidate_content)
            self.assertEqual((output / published_name).read_bytes(), published_content)
            self.assertEqual(verify_wheel_assets([candidate, published], output), prepared)

    def test_rejects_unowned_candidates_and_incomplete_final_inventory(self) -> None:
        package = PackageIdentity("inferlab-adapter-sdk", "0.6.2")
        filename = "inferlab_adapter_sdk-0.6.2-py3-none-any.whl"
        content = b"candidate"

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidates = root / "candidates"
            candidates.mkdir()
            (candidates / filename).write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            (candidates / f"{filename}.sha256").write_text(f"{digest}  {filename}\n")
            extra = "unowned_package-1.0.0-py3-none-any.whl"
            (candidates / extra).write_bytes(b"extra")
            (candidates / f"{extra}.sha256").write_text(
                f"{hashlib.sha256(b'extra').hexdigest()}  {extra}\n"
            )

            with self.assertRaises(ReleaseAssetError):
                prepare_assets(
                    [package],
                    root / "assets",
                    candidates,
                )

            final = root / "final"
            final.mkdir()
            (final / filename).write_bytes(content)
            with self.assertRaises(ReleaseAssetError):
                verify_wheel_assets([package], final)

    def test_verifies_downloaded_repository_and_aggregate_release_assets(self) -> None:
        package = PackageIdentity("inferlab-adapter-sdk", "0.6.2")
        wheel_name = "inferlab_adapter_sdk-0.6.2-py3-none-any.whl"
        wheel_content = b"qualified wheel"

        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary) / "repository"
            repository.mkdir()
            self._write_repository_assets(repository)
            self.assertEqual(len(verify_repository_assets(repository)), 9)

            aggregate = pathlib.Path(temporary) / "aggregate"
            aggregate.mkdir()
            self._write_repository_assets(aggregate)
            (aggregate / wheel_name).write_bytes(wheel_content)
            wheel_digest = hashlib.sha256(wheel_content).hexdigest()
            (aggregate / f"{wheel_name}.sha256").write_text(f"{wheel_digest}  {wheel_name}\n")
            self.assertEqual(len(verify_aggregate_assets([package], aggregate)), 11)

            (aggregate / "unexpected").write_text("extra")
            with self.assertRaises(ReleaseAssetError):
                verify_aggregate_assets([package], aggregate)


if __name__ == "__main__":
    unittest.main()
