#!/usr/bin/env python3
"""Pre-warm a JFrog remote cache for Hugging Face model files.

The default mode intentionally does not write model payloads to disk. It issues
GET requests through a JFrog remote, consumes each response body, and discards
the bytes locally so the remote cache can be populated without using laptop
storage for model shards.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import dataclasses
import fnmatch
import json
import json.decoder
import os
import shutil
import ssl
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_ID = "nvidia/GLM-5.2-NVFP4"
DEFAULT_REVISION = "main"
DEFAULT_STATE_FILE = ".hf-jfrog-prewarm-state.json"
DEFAULT_CHUNK_SIZE = 16 * 1024 * 1024
DEFAULT_MAX_INDEX_BYTES = 256 * 1024 * 1024
DEFAULT_INDEX_FILENAME = "model.safetensors.index.json"
RUNTIME_PATTERNS = (
    "model-*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


@dataclasses.dataclass(frozen=True)
class ModelFile:
    name: str
    size: int | None = None


@dataclasses.dataclass(frozen=True)
class AuthConfig:
    bearer_token: str | None = None
    basic_user: str | None = None
    basic_password: str | None = None


@dataclasses.dataclass(frozen=True)
class TlsConfig:
    ca_bundle: str | None = None
    insecure_skip_verify: bool = False


@dataclasses.dataclass(frozen=True)
class DownloadResult:
    file: ModelFile
    bytes_read: int
    elapsed_seconds: float
    skipped: bool = False
    error: str | None = None


class ProgressTracker:
    def __init__(self, files: list[ModelFile], pending: list[ModelFile], interval_seconds: float) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._interval_seconds = interval_seconds
        self._total_files = len(files)
        self._pending_files = len(pending)
        self._known_total_bytes = sum(file.size for file in pending if file.size is not None)
        self._completed_files = 0
        self._failed_files = 0
        self._bytes_read = 0
        self._active: dict[str, int] = {}
        self._started_at = time.monotonic()

    def start(self) -> None:
        if self._interval_seconds <= 0:
            return
        self._thread = threading.Thread(target=self._run, name="progress-reporter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 1)
        self.report(final=True)

    def file_started(self, file: ModelFile) -> None:
        with self._lock:
            self._active[file.name] = 0

    def add_bytes(self, file: ModelFile, amount: int) -> None:
        with self._lock:
            self._bytes_read += amount
            self._active[file.name] = self._active.get(file.name, 0) + amount

    def file_completed(self, file: ModelFile) -> None:
        with self._lock:
            self._completed_files += 1
            self._active.pop(file.name, None)

    def file_failed(self, file: ModelFile) -> None:
        with self._lock:
            self._failed_files += 1
            self._active.pop(file.name, None)

    def file_inactive(self, file: ModelFile) -> None:
        with self._lock:
            self._active.pop(file.name, None)

    def report(self, final: bool = False) -> None:
        with self._lock:
            elapsed = max(time.monotonic() - self._started_at, 0.001)
            rate = int(self._bytes_read / elapsed)
            active = sorted(self._active.items())[:3]
            known_total = self._known_total_bytes
            if known_total:
                percent = min(self._bytes_read / known_total * 100, 999.9)
                total_text = f"{human_size(self._bytes_read)} / {human_size(known_total)} ({percent:.1f}%)"
            else:
                total_text = f"{human_size(self._bytes_read)} / unknown"
            active_text = ", ".join(f"{name}: {human_size(size)}" for name, size in active) or "none"
            prefix = "FINAL" if final else "PROGRESS"
            message = (
                f"{prefix} files {self._completed_files}/{self._pending_files} complete, "
                f"{self._failed_files} failed, bytes {total_text}, rate {human_size(rate)}/s, "
                f"active: {active_text}"
            )
        print(message, flush=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.report()


def parse_size(value: str) -> int:
    text = value.strip().lower()
    units = {
        "b": 1,
        "k": 1024,
        "ki": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mi": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gi": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            return int(float(number) * multiplier)
    return int(text)


def human_size(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def build_resolve_url(base_url: str, repo_id: str, revision: str, filename: str) -> str:
    base = normalize_base_url(base_url)
    repo_path = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    revision_path = urllib.parse.quote(revision, safe="")
    file_path = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    return f"{base}/{repo_path}/resolve/{revision_path}/{file_path}"


def build_api_url(base_url: str, repo_id: str) -> str:
    base = normalize_base_url(base_url)
    repo_path = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    return f"{base}/api/models/{repo_path}?blobs=true"


def host_of(url: str) -> str | None:
    return urllib.parse.urlparse(url).netloc.lower() or None


def ensure_redirect_stayed_on_jfrog(original_url: str, final_url: str, allow_external_redirect: bool) -> None:
    if allow_external_redirect:
        return
    original_host = host_of(original_url)
    final_host = host_of(final_url)
    if original_host and final_host and original_host != final_host:
        raise RuntimeError(
            "request was redirected outside JFrog "
            f"({original_host} -> {final_host}); this would not reliably warm the JFrog cache"
        )


def auth_headers(auth: AuthConfig) -> dict[str, str]:
    headers: dict[str, str] = {"User-Agent": "hf-jfrog-prewarm/1.0"}
    if auth.bearer_token:
        headers["Authorization"] = f"Bearer {auth.bearer_token}"
    elif auth.basic_user is not None and auth.basic_password is not None:
        raw = f"{auth.basic_user}:{auth.basic_password}".encode("utf-8")
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")
    return headers


def create_ssl_context(tls: TlsConfig) -> ssl.SSLContext | None:
    if tls.insecure_skip_verify:
        return ssl._create_unverified_context()
    if tls.ca_bundle:
        return ssl.create_default_context(cafile=tls.ca_bundle)
    return None


def request_json(url: str, auth: AuthConfig, timeout: float, tls: TlsConfig) -> dict:
    request = urllib.request.Request(url, headers=auth_headers(auth))
    with urllib.request.urlopen(request, timeout=timeout, context=create_ssl_context(tls)) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read()
        text = body.decode(charset, errors="replace")
        try:
            return json.loads(text)
        except json.decoder.JSONDecodeError as exc:
            status = getattr(response, "status", "unknown")
            content_type = response.headers.get("content-type", "unknown")
            snippet = text[:500].replace("\n", "\\n") or "<empty>"
            raise RuntimeError(
                f"expected JSON from {url}, got status {status}, "
                f"content-type {content_type}, body starts with {snippet!r}"
            ) from exc


def request_bytes(
    url: str,
    auth: AuthConfig,
    timeout: float,
    tls: TlsConfig,
    max_bytes: int,
    allow_external_redirect: bool,
) -> bytes:
    request = urllib.request.Request(url, headers=auth_headers(auth))
    with urllib.request.urlopen(request, timeout=timeout, context=create_ssl_context(tls)) as response:
        ensure_redirect_stayed_on_jfrog(url, response.geturl(), allow_external_redirect)
        chunks = []
        total = 0
        while True:
            chunk = response.read(min(DEFAULT_CHUNK_SIZE, max_bytes + 1 - total))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(f"response from {url} exceeded max index size {human_size(max_bytes)}")


def runtime_file_from_sibling(sibling: dict) -> ModelFile | None:
    name = sibling.get("rfilename")
    if not isinstance(name, str):
        return None
    if not any(fnmatch.fnmatchcase(name, pattern) for pattern in RUNTIME_PATTERNS):
        return None
    size = sibling.get("size")
    return ModelFile(name=name, size=size if isinstance(size, int) else None)


def discover_files_from_api(base_url: str, repo_id: str, auth: AuthConfig, timeout: float, tls: TlsConfig) -> list[ModelFile]:
    payload = request_json(build_api_url(base_url, repo_id), auth, timeout, tls)
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError("model API response did not include a siblings list")
    files = [file for item in siblings if (file := runtime_file_from_sibling(item)) is not None]
    if not files:
        raise RuntimeError("model API response did not include any runtime-minimal files")
    return sorted(files, key=lambda item: item.name)


def files_from_index_payload(payload: dict, index_name: str, index_size: int | None = None) -> list[ModelFile]:
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("index payload does not contain a valid weight_map")
    shard_names = sorted({value for value in weight_map.values() if isinstance(value, str)})
    if not shard_names:
        raise RuntimeError("index payload did not include shard filenames")
    files = [ModelFile(name=name) for name in shard_names]
    files.append(ModelFile(name=index_name, size=index_size))
    for name in (
        "config.json",
        "generation_config.json",
        "hf_quant_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        files.append(ModelFile(name=name))
    return sorted(files, key=lambda item: item.name)


def discover_files_from_remote_index(
    base_url: str,
    repo_id: str,
    revision: str,
    index_name: str,
    auth: AuthConfig,
    timeout: float,
    tls: TlsConfig,
    max_index_bytes: int,
    allow_external_redirect: bool,
) -> list[ModelFile]:
    url = build_resolve_url(base_url, repo_id, revision, index_name)
    body = request_bytes(url, auth, timeout, tls, max_index_bytes, allow_external_redirect)
    payload = json.loads(body.decode("utf-8"))
    return files_from_index_payload(payload, index_name, len(body))


def metadata_auth_for(base_url: str, metadata_base_url: str, jfrog_auth: AuthConfig, hf_token: str | None) -> AuthConfig:
    if host_of(base_url) == host_of(metadata_base_url):
        return jfrog_auth
    return AuthConfig(bearer_token=hf_token)


def discover_shards_from_index(path: Path) -> list[ModelFile]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return files_from_index_payload(payload, path.name, path.stat().st_size)


def load_state(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"completed": {}}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {"completed": {}}
    completed = payload.get("completed")
    if not isinstance(completed, dict):
        payload["completed"] = {}
    return payload


def save_state(path: Path | None, state: dict) -> None:
    if path is None:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def is_completed(state: dict, file: ModelFile) -> bool:
    completed = state.get("completed", {})
    entry = completed.get(file.name) if isinstance(completed, dict) else None
    if not isinstance(entry, dict):
        return False
    if file.size is None:
        return entry.get("bytes_read", 0) > 0
    return entry.get("bytes_read") == file.size


def mark_completed(state: dict, file: ModelFile, bytes_read: int) -> None:
    state.setdefault("completed", {})[file.name] = {
        "bytes_read": bytes_read,
        "expected_size": file.size,
        "completed_at": int(time.time()),
    }


def ensure_temp_capacity(temp_dir: Path, files: Iterable[ModelFile], workers: int) -> None:
    known_sizes = [file.size for file in files if file.size is not None]
    if not known_sizes:
        raise RuntimeError("temp-file mode requires known file sizes from metadata")
    required = max(known_sizes) * max(workers, 1) + 1024**3
    free = shutil.disk_usage(temp_dir).free
    if free < required:
        raise RuntimeError(
            f"not enough free space in {temp_dir}: need {human_size(required)}, have {human_size(free)}"
        )


def consume_response(response, file: ModelFile, chunk_size: int, mode: str, temp_dir: Path, progress: ProgressTracker | None) -> int:
    bytes_read = 0
    temp_handle = None
    temp_name = None
    try:
        if mode == "temp-file":
            temp_dir.mkdir(parents=True, exist_ok=True)
            safe_prefix = file.name.replace("/", "_")
            fd, temp_name = tempfile.mkstemp(prefix=f"{safe_prefix}.", suffix=".tmp", dir=temp_dir)
            temp_handle = os.fdopen(fd, "wb")
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                return bytes_read
            if temp_handle is not None:
                temp_handle.write(chunk)
            bytes_read += len(chunk)
            if progress is not None:
                progress.add_bytes(file, len(chunk))
    finally:
        if temp_handle is not None:
            temp_handle.close()
        if temp_name is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name)


def download_once(
    file: ModelFile,
    base_url: str,
    repo_id: str,
    revision: str,
    auth: AuthConfig,
    timeout: float,
    chunk_size: int,
    mode: str,
    temp_dir: Path,
    allow_external_redirect: bool,
    tls: TlsConfig,
    progress: ProgressTracker | None,
) -> DownloadResult:
    start = time.monotonic()
    url = build_resolve_url(base_url, repo_id, revision, file.name)
    request = urllib.request.Request(url, headers=auth_headers(auth))
    if progress is not None:
        progress.file_started(file)
    with urllib.request.urlopen(request, timeout=timeout, context=create_ssl_context(tls)) as response:
        ensure_redirect_stayed_on_jfrog(url, response.geturl(), allow_external_redirect)
        bytes_read = consume_response(response, file, chunk_size, mode, temp_dir, progress)
    elapsed = time.monotonic() - start
    if file.size is not None and bytes_read != file.size:
        raise RuntimeError(f"received {bytes_read} bytes, expected {file.size}")
    return DownloadResult(file=file, bytes_read=bytes_read, elapsed_seconds=elapsed)


def download_with_retries(
    file: ModelFile,
    base_url: str,
    repo_id: str,
    revision: str,
    auth: AuthConfig,
    timeout: float,
    chunk_size: int,
    mode: str,
    temp_dir: Path,
    retries: int,
    allow_external_redirect: bool,
    tls: TlsConfig,
    progress: ProgressTracker | None,
) -> DownloadResult:
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            return download_once(
                file,
                base_url,
                repo_id,
                revision,
                auth,
                timeout,
                chunk_size,
                mode,
                temp_dir,
                allow_external_redirect,
                tls,
                progress,
            )
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
            if progress is not None:
                progress.file_inactive(file)
            last_error = str(exc)
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 30))
    return DownloadResult(file=file, bytes_read=0, elapsed_seconds=0.0, error=last_error or "unknown error")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-warm Public JFrog for Hugging Face model runtime files.")
    parser.add_argument("--jfrog-base-url", required=True, help="Public JFrog remote base URL mapped to huggingface.co.")
    parser.add_argument(
        "-r",
        "--repo-id",
        default=os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID),
        help=f"Hugging Face repo id. Defaults to HF_REPO_ID or {DEFAULT_REPO_ID}.",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--chunk-size", type=parse_size, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between progress updates while downloads are running. Set 0 to disable periodic updates.",
    )
    parser.add_argument("--mode", choices=("discard", "temp-file"), default="discard")
    parser.add_argument("--temp-dir", default=tempfile.gettempdir())
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--force", action="store_true", help="Do not skip files marked complete in the state file.")
    parser.add_argument("--dry-run", action="store_true", help="List target files without downloading response bodies.")
    parser.add_argument("--index-file", type=Path, default=Path(DEFAULT_INDEX_FILENAME))
    parser.add_argument(
        "--index-filename",
        default=DEFAULT_INDEX_FILENAME,
        help=f"Remote Hugging Face index filename used by remote-index discovery. Defaults to {DEFAULT_INDEX_FILENAME}.",
    )
    parser.add_argument("--no-api-discovery", action="store_true", help="Use the local index file instead of JFrog API metadata.")
    parser.add_argument(
        "--discovery",
        choices=("auto", "api", "remote-index", "local-index"),
        default="auto",
        help="How to discover runtime files. auto tries API, then remote index via JFrog, then local index.",
    )
    parser.add_argument("--max-index-bytes", type=parse_size, default=DEFAULT_MAX_INDEX_BYTES)
    parser.add_argument(
        "--metadata-base-url",
        default=os.environ.get("HF_METADATA_BASE_URL"),
        help="Base URL for Hugging Face model metadata discovery. Defaults to --jfrog-base-url. Use https://huggingface.co if JFrog does not proxy /api/models.",
    )
    parser.add_argument(
        "--allow-external-redirect",
        action="store_true",
        help="Allow redirects away from the JFrog host. This may bypass JFrog cache warming.",
    )
    parser.add_argument("--jfrog-token", default=os.environ.get("JFROG_TOKEN"))
    parser.add_argument("--jfrog-user", default=os.environ.get("JFROG_USER"))
    parser.add_argument("--jfrog-password", default=os.environ.get("JFROG_PASSWORD"))
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token used only when --metadata-base-url points outside the JFrog host.",
    )
    parser.add_argument(
        "--ca-bundle",
        default=os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE"),
        help="Custom CA bundle path for JFrog TLS verification. Defaults to SSL_CERT_FILE, REQUESTS_CA_BUNDLE, or CURL_CA_BUNDLE.",
    )
    parser.add_argument(
        "--insecure-skip-tls-verify",
        action="store_true",
        help="Disable TLS certificate verification. Use only for temporary diagnosis.",
    )
    return parser.parse_args(argv)


def print_file_plan(files: list[ModelFile]) -> None:
    total_known = sum(file.size for file in files if file.size is not None)
    unknown_count = sum(1 for file in files if file.size is None)
    print(f"Target files: {len(files)}")
    print(f"Known total size: {human_size(total_known)}")
    if unknown_count:
        print(f"Unknown-size files: {unknown_count}")
    for file in files:
        print(f"  {file.name} ({human_size(file.size)})")


def discover_files(args: argparse.Namespace, auth: AuthConfig, metadata_auth: AuthConfig, tls: TlsConfig) -> list[ModelFile]:
    metadata_base_url = args.metadata_base_url or args.jfrog_base_url
    if args.no_api_discovery:
        return discover_shards_from_index(args.index_file)
    if args.discovery == "api":
        return discover_files_from_api(metadata_base_url, args.repo_id, metadata_auth, args.timeout, tls)
    if args.discovery == "remote-index":
        return discover_files_from_remote_index(
            args.jfrog_base_url,
            args.repo_id,
            args.revision,
            args.index_filename,
            auth,
            args.timeout,
            tls,
            args.max_index_bytes,
            args.allow_external_redirect,
        )
    if args.discovery == "local-index":
        return discover_shards_from_index(args.index_file)

    errors = []
    try:
        return discover_files_from_api(metadata_base_url, args.repo_id, metadata_auth, args.timeout, tls)
    except Exception as exc:
        errors.append(f"api: {exc}")
        print(f"API discovery failed: {exc}", file=sys.stderr)
    try:
        return discover_files_from_remote_index(
            args.jfrog_base_url,
            args.repo_id,
            args.revision,
            args.index_filename,
            auth,
            args.timeout,
            tls,
            args.max_index_bytes,
            args.allow_external_redirect,
        )
    except Exception as exc:
        errors.append(f"remote-index: {exc}")
        print(f"Remote index discovery failed: {exc}", file=sys.stderr)
    try:
        print(f"Falling back to local index file: {args.index_file}", file=sys.stderr)
        return discover_shards_from_index(args.index_file)
    except Exception as exc:
        errors.append(f"local-index: {exc}")
        raise RuntimeError("; ".join(errors)) from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    auth = AuthConfig(args.jfrog_token, args.jfrog_user, args.jfrog_password)
    metadata_base_url = args.metadata_base_url or args.jfrog_base_url
    metadata_auth = metadata_auth_for(args.jfrog_base_url, metadata_base_url, auth, args.hf_token)
    tls = TlsConfig(args.ca_bundle, args.insecure_skip_tls_verify)
    if tls.insecure_skip_verify:
        print("WARNING: TLS certificate verification is disabled.", file=sys.stderr)
    try:
        files = discover_files(args, auth, metadata_auth, tls)
    except Exception as exc:
        print(f"File discovery failed: {exc}", file=sys.stderr)
        return 2

    print_file_plan(files)
    if args.dry_run:
        print("Dry run complete. No response bodies were downloaded.")
        return 0

    print(f"Mode: {args.mode}")
    if args.mode == "discard":
        print("Response bodies will be consumed and discarded. Model payloads will not be written locally.")
    else:
        temp_dir = Path(args.temp_dir)
        try:
            ensure_temp_capacity(temp_dir, files, args.workers)
        except Exception as exc:
            print(f"Temp-file mode refused: {exc}", file=sys.stderr)
            return 2
        print(f"Temporary payloads will be written under {temp_dir} and deleted after each file completes.")

    state_path = None if args.no_state else Path(args.state_file)
    state = load_state(state_path)
    pending = [file for file in files if args.force or not is_completed(state, file)]
    skipped = len(files) - len(pending)
    if skipped:
        print(f"Skipping {skipped} files already marked complete in {state_path}.")
    if not pending:
        print("Nothing to do.")
        return 0

    failures: list[DownloadResult] = []
    completed_bytes = 0
    started = time.monotonic()
    progress = ProgressTracker(files, pending, args.progress_interval)
    progress.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_with_retries,
                file,
                args.jfrog_base_url,
                args.repo_id,
                args.revision,
                auth,
                args.timeout,
                args.chunk_size,
                args.mode,
                Path(args.temp_dir),
                args.retries,
                args.allow_external_redirect,
                tls,
                progress,
            ): file
            for file in pending
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result.error:
                progress.file_failed(result.file)
                failures.append(result)
                print(f"FAIL {result.file.name}: {result.error}", file=sys.stderr)
                continue
            completed_bytes += result.bytes_read
            progress.file_completed(result.file)
            rate = result.bytes_read / result.elapsed_seconds if result.elapsed_seconds > 0 else 0
            mark_completed(state, result.file, result.bytes_read)
            save_state(state_path, state)
            print(
                f"OK {result.file.name}: {human_size(result.bytes_read)} "
                f"in {result.elapsed_seconds:.1f}s ({human_size(int(rate))}/s)"
            )
    progress.stop()

    elapsed = time.monotonic() - started
    print(f"Completed bytes this run: {human_size(completed_bytes)} in {elapsed:.1f}s")
    if failures:
        print(f"Failed files: {len(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
