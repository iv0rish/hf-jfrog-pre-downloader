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
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "hf_quant_config.json",
                "model-00001-of-00047.safetensors",
                "model-00002-of-00047.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
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

    def test_metadata_base_url_flag(self):
        args = prewarm.parse_args(
            [
                "--jfrog-base-url",
                "https://jfrog.example/artifactory/hf",
                "--metadata-base-url",
                "https://huggingface.co",
            ]
        )

        self.assertEqual(args.metadata_base_url, "https://huggingface.co")

    def test_discovery_remote_index_flag(self):
        args = prewarm.parse_args(
            [
                "--jfrog-base-url",
                "https://jfrog.example/artifactory/hf",
                "--discovery",
                "remote-index",
            ]
        )

        self.assertEqual(args.discovery, "remote-index")

    def test_metadata_auth_uses_hf_token_for_different_host(self):
        jfrog_auth = prewarm.AuthConfig(bearer_token="jfrog-token")

        auth = prewarm.metadata_auth_for(
            "https://jfrog.example/artifactory/hf",
            "https://huggingface.co",
            jfrog_auth,
            "hf-token",
        )

        self.assertEqual(auth.bearer_token, "hf-token")

    def test_metadata_auth_uses_jfrog_auth_for_same_host(self):
        jfrog_auth = prewarm.AuthConfig(bearer_token="jfrog-token")

        auth = prewarm.metadata_auth_for(
            "https://jfrog.example/artifactory/hf",
            "https://jfrog.example/artifactory/hf",
            jfrog_auth,
            "hf-token",
        )

        self.assertIs(auth, jfrog_auth)

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

    def test_request_json_reports_non_json_body(self):
        class Headers(dict):
            def get_content_charset(self):
                return "utf-8"

        class Response:
            status = 200
            headers = Headers({"content-type": "text/html"})

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"<html>login</html>"

        with mock.patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "expected JSON"):
                prewarm.request_json(
                    "https://jfrog.example/artifactory/hf/api/models/org/model?blobs=true",
                    prewarm.AuthConfig(),
                    1,
                    prewarm.TlsConfig(),
                )

    def test_files_from_index_payload_adds_runtime_sidecars(self):
        files = prewarm.files_from_index_payload(
            {
                "weight_map": {
                    "a": "model-00002-of-00047.safetensors",
                    "b": "model-00001-of-00047.safetensors",
                }
            },
            "model.safetensors.index.json",
            123,
        )

        self.assertIn(prewarm.ModelFile("model-00001-of-00047.safetensors"), files)
        self.assertIn(prewarm.ModelFile("tokenizer.json"), files)
        self.assertIn(prewarm.ModelFile("model.safetensors.index.json", 123), files)


if __name__ == "__main__":
    unittest.main()
