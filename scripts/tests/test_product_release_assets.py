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
)


class ProductReleaseAssetTests(unittest.TestCase):
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
            output = pathlib.Path(temporary) / "assets"
            prepared = prepare_assets(packages, output, read_json, read_bytes)

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
                output = pathlib.Path(temporary) / "assets"
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
                        read_case_json,
                        lambda _url: content,
                    )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
