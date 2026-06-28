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
import os
import shutil
import sys
import tempfile
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
class DownloadResult:
    file: ModelFile
    bytes_read: int
    elapsed_seconds: float
    skipped: bool = False
    error: str | None = None


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


def request_json(url: str, auth: AuthConfig, timeout: float) -> dict:
    request = urllib.request.Request(url, headers=auth_headers(auth))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset))


def runtime_file_from_sibling(sibling: dict) -> ModelFile | None:
    name = sibling.get("rfilename")
    if not isinstance(name, str):
        return None
    if not any(fnmatch.fnmatchcase(name, pattern) for pattern in RUNTIME_PATTERNS):
        return None
    size = sibling.get("size")
    return ModelFile(name=name, size=size if isinstance(size, int) else None)


def discover_files_from_api(base_url: str, repo_id: str, auth: AuthConfig, timeout: float) -> list[ModelFile]:
    payload = request_json(build_api_url(base_url, repo_id), auth, timeout)
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError("model API response did not include a siblings list")
    files = [file for item in siblings if (file := runtime_file_from_sibling(item)) is not None]
    if not files:
        raise RuntimeError("model API response did not include any runtime-minimal files")
    return sorted(files, key=lambda item: item.name)


def discover_shards_from_index(path: Path) -> list[ModelFile]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"{path} does not contain a valid weight_map")
    names = sorted({value for value in weight_map.values() if isinstance(value, str)})
    files = [ModelFile(name=name) for name in names]
    if path.name not in names:
        files.append(ModelFile(name=path.name, size=path.stat().st_size))
    return sorted(files, key=lambda item: item.name)


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


def consume_to_discard(response, chunk_size: int) -> int:
    bytes_read = 0
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            return bytes_read
        bytes_read += len(chunk)


def consume_to_temp_file(response, file: ModelFile, temp_dir: Path, chunk_size: int) -> int:
    bytes_read = 0
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = file.name.replace("/", "_")
    fd, temp_name = tempfile.mkstemp(prefix=f"{safe_prefix}.", suffix=".tmp", dir=temp_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_read += len(chunk)
        return bytes_read
    finally:
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
) -> DownloadResult:
    start = time.monotonic()
    url = build_resolve_url(base_url, repo_id, revision, file.name)
    request = urllib.request.Request(url, headers=auth_headers(auth))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        ensure_redirect_stayed_on_jfrog(url, response.geturl(), allow_external_redirect)
        if mode == "temp-file":
            bytes_read = consume_to_temp_file(response, file, temp_dir, chunk_size)
        else:
            bytes_read = consume_to_discard(response, chunk_size)
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
            )
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
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
    parser.add_argument("--mode", choices=("discard", "temp-file"), default="discard")
    parser.add_argument("--temp-dir", default=tempfile.gettempdir())
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    parser.add_argument("--no-state", action="store_true")
    parser.add_argument("--force", action="store_true", help="Do not skip files marked complete in the state file.")
    parser.add_argument("--dry-run", action="store_true", help="List target files without downloading response bodies.")
    parser.add_argument("--index-file", type=Path, default=Path("model.safetensors.index.json"))
    parser.add_argument("--no-api-discovery", action="store_true", help="Use the local index file instead of JFrog API metadata.")
    parser.add_argument(
        "--allow-external-redirect",
        action="store_true",
        help="Allow redirects away from the JFrog host. This may bypass JFrog cache warming.",
    )
    parser.add_argument("--jfrog-token", default=os.environ.get("JFROG_TOKEN"))
    parser.add_argument("--jfrog-user", default=os.environ.get("JFROG_USER"))
    parser.add_argument("--jfrog-password", default=os.environ.get("JFROG_PASSWORD"))
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    auth = AuthConfig(args.jfrog_token, args.jfrog_user, args.jfrog_password)
    try:
        if args.no_api_discovery:
            files = discover_shards_from_index(args.index_file)
        else:
            try:
                files = discover_files_from_api(args.jfrog_base_url, args.repo_id, auth, args.timeout)
            except Exception as exc:
                print(f"API discovery failed: {exc}", file=sys.stderr)
                print(f"Falling back to local index file: {args.index_file}", file=sys.stderr)
                files = discover_shards_from_index(args.index_file)
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
            ): file
            for file in pending
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result.error:
                failures.append(result)
                print(f"FAIL {result.file.name}: {result.error}", file=sys.stderr)
                continue
            completed_bytes += result.bytes_read
            rate = result.bytes_read / result.elapsed_seconds if result.elapsed_seconds > 0 else 0
            mark_completed(state, result.file, result.bytes_read)
            save_state(state_path, state)
            print(
                f"OK {result.file.name}: {human_size(result.bytes_read)} "
                f"in {result.elapsed_seconds:.1f}s ({human_size(int(rate))}/s)"
            )

    elapsed = time.monotonic() - started
    print(f"Completed bytes this run: {human_size(completed_bytes)} in {elapsed:.1f}s")
    if failures:
        print(f"Failed files: {len(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
