import tempfile
import unittest
import ssl
from pathlib import Path
from unittest import mock

import hf_jfrog_prewarm as prewarm


class PrewarmTests(unittest.TestCase):
    def test_build_resolve_url_quotes_path_parts(self):
        url = prewarm.build_resolve_url(
            "https://jfrog.example/artifactory/hf/",
            "org/model name",
            "main",
            "nested/file name.json",
        )

        self.assertEqual(
            url,
            "https://jfrog.example/artifactory/hf/org/model%20name/resolve/main/nested/file%20name.json",
        )

    def test_runtime_file_filtering(self):
        siblings = [
            {"rfilename": "model-00001-of-00047.safetensors", "size": 10},
            {"rfilename": "README.md", "size": 20},
            {"rfilename": "tokenizer.json", "size": 30},
        ]

        files = [prewarm.runtime_file_from_sibling(item) for item in siblings]

        self.assertEqual(
            [file.name for file in files if file is not None],
            ["model-00001-of-00047.safetensors", "tokenizer.json"],
        )

    def test_state_marks_file_complete_only_when_size_matches(self):
        state = {"completed": {"file.bin": {"bytes_read": 10}}}

        self.assertTrue(prewarm.is_completed(state, prewarm.ModelFile("file.bin", 10)))
        self.assertFalse(prewarm.is_completed(state, prewarm.ModelFile("file.bin", 11)))
        self.assertFalse(prewarm.is_completed(state, prewarm.ModelFile("other.bin", 10)))

    def test_discover_shards_from_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "model.safetensors.index.json"
            index_path.write_text(
                """
                {
                  "weight_map": {
                    "a": "model-00002-of-00047.safetensors",
                    "b": "model-00001-of-00047.safetensors"
                  }
                }
                """,
                encoding="utf-8",
            )

            files = prewarm.discover_shards_from_index(index_path)

        self.assertEqual(
            [file.name for file in files],
            [
                "model-00001-of-00047.safetensors",
                "model-00002-of-00047.safetensors",
                "model.safetensors.index.json",
            ],
        )

    def test_temp_capacity_rejects_insufficient_space(self):
        files = [prewarm.ModelFile("large.bin", 100)]
        usage = mock.Mock(free=50)

        with mock.patch("shutil.disk_usage", return_value=usage):
            with self.assertRaisesRegex(RuntimeError, "not enough free space"):
                prewarm.ensure_temp_capacity(Path("/tmp"), files, workers=1)

    def test_parse_size(self):
        self.assertEqual(prewarm.parse_size("16MiB"), 16 * 1024 * 1024)
        self.assertEqual(prewarm.parse_size("1g"), 1024**3)
        self.assertEqual(prewarm.parse_size("512"), 512)

    def test_repo_id_flag(self):
        args = prewarm.parse_args(
            [
                "--jfrog-base-url",
                "https://jfrog.example/artifactory/hf",
                "--repo-id",
                "org/custom-model",
            ]
        )

        self.assertEqual(args.repo_id, "org/custom-model")

    def test_repo_id_short_flag(self):
        args = prewarm.parse_args(
            [
                "--jfrog-base-url",
                "https://jfrog.example/artifactory/hf",
                "-r",
                "org/custom-model",
            ]
        )

        self.assertEqual(args.repo_id, "org/custom-model")

    def test_ca_bundle_flag(self):
        args = prewarm.parse_args(
            [
                "--jfrog-base-url",
                "https://jfrog.example/artifactory/hf",
                "--ca-bundle",
                "/tmp/company-ca.pem",
            ]
        )

        self.assertEqual(args.ca_bundle, "/tmp/company-ca.pem")

    def test_insecure_tls_context_disables_verification(self):
        context = prewarm.create_ssl_context(prewarm.TlsConfig(insecure_skip_verify=True))

        self.assertEqual(context.verify_mode, ssl.CERT_NONE)
        self.assertFalse(context.check_hostname)

    def test_external_redirect_is_rejected_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "redirected outside JFrog"):
            prewarm.ensure_redirect_stayed_on_jfrog(
                "https://jfrog.example/artifactory/hf/org/model/resolve/main/file.bin",
                "https://cdn.hf.co/file.bin",
                allow_external_redirect=False,
            )

    def test_external_redirect_can_be_allowed(self):
        prewarm.ensure_redirect_stayed_on_jfrog(
            "https://jfrog.example/artifactory/hf/org/model/resolve/main/file.bin",
            "https://cdn.hf.co/file.bin",
            allow_external_redirect=True,
        )


if __name__ == "__main__":
    unittest.main()
