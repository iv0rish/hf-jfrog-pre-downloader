# Hugging Face JFrog Pre-Downloader

Diskless pre-warmer for Hugging Face model files through a Public JFrog remote
repository.

The target use case is:

```text
Hugging Face -> Public JFrog -> Private JFrog -> GPU Instance
```

When a GPU instance requests a large model for the first time, both JFrog layers
may need to fetch and cache the same large files. This script runs from a
developer laptop that can access Public JFrog and asks Public JFrog to fetch the
model files first.

## Important Behavior

- The default mode does **not** save model payloads on the laptop.
- The script performs full `GET` requests and reads the response body in chunks.
- Each chunk is immediately discarded in `--mode discard`.
- The laptop still receives the network stream. This avoids local storage use,
  not network transfer.
- A tiny JSON state file is written by default so completed files can be skipped
  on later runs.

## Usage

Dry-run the default model:

```bash
python3 hf_jfrog_prewarm.py \
  --jfrog-base-url https://public-jfrog.example.com/artifactory/huggingface-remote \
  --dry-run
```

Pre-warm using the diskless discard mode:

```bash
python3 hf_jfrog_prewarm.py \
  --jfrog-base-url https://public-jfrog.example.com/artifactory/huggingface-remote \
  --repo-id nvidia/GLM-5.2-NVFP4 \
  --workers 2
```

`--repo-id` can be any Hugging Face model repository id that the JFrog remote
can access. You can also set it with `HF_REPO_ID`.

While downloads are running, the script prints periodic progress updates every
30 seconds by default. Adjust this with `--progress-interval 10`, or disable
periodic updates with `--progress-interval 0`.

Use JFrog authentication:

```bash
JFROG_TOKEN=... python3 hf_jfrog_prewarm.py \
  --jfrog-base-url https://public-jfrog.example.com/artifactory/huggingface-remote
```

Use a corporate or private CA bundle when Python cannot verify the JFrog
certificate:

```bash
python3 hf_jfrog_prewarm.py \
  --jfrog-base-url https://public-jfrog.example.com/artifactory/huggingface-remote \
  --ca-bundle /path/to/company-ca-bundle.pem \
  --dry-run
```

The same path can be supplied with `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, or
`CURL_CA_BUNDLE`. For temporary diagnosis only, TLS verification can be disabled:

```bash
python3 hf_jfrog_prewarm.py \
  --jfrog-base-url https://public-jfrog.example.com/artifactory/huggingface-remote \
  --insecure-skip-tls-verify \
  --dry-run
```

If API discovery fails with an error like `expected JSON`, the JFrog remote is
probably not proxying Hugging Face's `/api/models/...` endpoint as JSON. The
default `--discovery auto` mode will then try to fetch
`model.safetensors.index.json` through JFrog and parse the shard list in memory.
This does not require direct laptop access to Hugging Face.

You can force that mode explicitly:

```bash
python3 hf_jfrog_prewarm.py \
  --jfrog-base-url https://public-jfrog.example.com/artifactory/huggingface-remote \
  --discovery remote-index \
  --insecure-skip-tls-verify \
  --dry-run
```

Use `--metadata-base-url https://huggingface.co` only when the laptop can access
Hugging Face directly. For private or gated metadata, set `HF_TOKEN` or pass
`--hf-token`. JFrog credentials are still used for downloads through
`--jfrog-base-url`.

Use temporary files instead of discard mode:

```bash
python3 hf_jfrog_prewarm.py \
  --jfrog-base-url https://public-jfrog.example.com/artifactory/huggingface-remote \
  --mode temp-file \
  --temp-dir /path/with/free/space \
  --workers 1
```

`temp-file` mode deletes each temporary payload after that file finishes, but it
still requires enough free disk for the largest shard times the worker count,
plus safety margin. For `nvidia/GLM-5.2-NVFP4`, most shards are about 10 GB.

## Target Files

By default the script targets runtime-minimal files for:

- `nvidia/GLM-5.2-NVFP4`
- revision `main`
- `model-*.safetensors`
- `model.safetensors.index.json`
- `config.json`, `generation_config.json`, `hf_quant_config.json`
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`

The URL shape is:

```text
{jfrog-base-url}/{repo-id}/resolve/{revision}/{filename}
```

The Public JFrog remote is expected to map its root to `https://huggingface.co`.
By default, the script rejects downloads that redirect away from the JFrog host.
If a request follows a Hugging Face CDN redirect directly from the laptop, it may
download bytes without warming the JFrog cache. Use `--allow-external-redirect`
only when that behavior is intentional.

## Notes

`HEAD` requests and tiny range requests are not used as the default warming
strategy because they should not be assumed to populate a complete remote-cache
artifact. The script consumes the complete response body so Public JFrog has a
chance to cache the complete file.
