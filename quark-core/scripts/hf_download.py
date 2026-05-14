#!/usr/bin/env python3
"""Quark HuggingFace dataset downloader with parallel chunk downloading.

Downloads each dataset shard using multiple parallel HTTP Range-request
connections (like Free Download Manager / IDM), then converts Parquet→JSONL.

Protocol on stdout:
  LOG:<message>           — human-readable status line
  PROGRESS:<0..1>         — overall fraction (0.0–1.0)
  SPEED:<bytes_per_sec>   — current download speed (float, bytes/s)
  BYTES:<done>/<total>    — cumulative bytes downloaded / estimated total
  FILE:<basename>         — file currently being downloaded
  DONE                    — all work complete
  ERROR:<message>         — fatal error (also causes non-zero exit)
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── stdout protocol ───────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"LOG:{msg}", flush=True)

def emit_progress(p: float) -> None:
    print(f"PROGRESS:{p:.4f}", flush=True)

def emit_speed(bps: float) -> None:
    print(f"SPEED:{bps:.0f}", flush=True)

def emit_bytes(done: int, total: int) -> None:
    print(f"BYTES:{done}/{total}", flush=True)

def emit_file(name: str) -> None:
    print(f"FILE:{name}", flush=True)


# ── speed tracker ─────────────────────────────────────────────────────────────

class SpeedTracker:
    """Thread-safe sliding-window download speed estimator (4-second window)."""

    WINDOW_SEC = 4.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._events: list[tuple[float, int]] = []  # (monotonic_time, bytes)

    def add(self, n: int) -> None:
        t = time.monotonic()
        with self._lock:
            self._total += n
            self._events.append((t, n))

    def speed_bps(self) -> float:
        now = time.monotonic()
        cutoff = now - self.WINDOW_SEC
        with self._lock:
            self._events = [(t, b) for t, b in self._events if t >= cutoff]
            if not self._events:
                return 0.0
            window_bytes = sum(b for _, b in self._events)
            span = max(now - self._events[0][0], 0.1)
        return window_bytes / span

    def total(self) -> int:
        with self._lock:
            return self._total


def start_reporter(tracker: SpeedTracker, total_fn, stop: threading.Event) -> threading.Thread:
    """Emit SPEED/BYTES/PROGRESS every 0.5 s from a daemon thread."""
    def _run() -> None:
        while not stop.wait(0.5):
            emit_speed(tracker.speed_bps())
            done = tracker.total()
            total = total_fn()
            emit_bytes(done, total)
            if total > 0:
                emit_progress(min(done / total, 0.99))
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ── parallel chunk downloader ─────────────────────────────────────────────────

def _dl_chunk(session, url: str, byte_start: int, byte_end: int,
              path: str, tracker: SpeedTracker) -> None:
    """Download one byte range into path, feeding bytes into tracker."""
    resp = session.get(
        url,
        headers={"Range": f"bytes={byte_start}-{byte_end}"},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                tracker.add(len(chunk))


def parallel_download(session, url: str, dest: str,
                      n_workers: int, tracker: SpeedTracker) -> bool:
    """Download url → dest using n_workers parallel Range connections.

    Falls back to a single connection if the server doesn't support Range
    requests or the file is smaller than 1 MiB.
    Returns True on success.
    """
    # Probe file size and Range support via HEAD
    try:
        head = session.head(url, timeout=30, allow_redirects=True)
        total = int(head.headers.get("Content-Length", 0))
        ranges_ok = head.headers.get("Accept-Ranges", "none").lower() == "bytes"
    except Exception as e:
        log(f"HEAD failed ({e}) — single connection")
        total = 0
        ranges_ok = False

    use_parallel = ranges_ok and total >= 1_048_576 and n_workers > 1

    if not use_parallel:
        try:
            resp = session.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(65536):
                    if chunk:
                        f.write(chunk)
                        tracker.add(len(chunk))
            return True
        except Exception as e:
            log(f"Download error: {e}")
            return False

    # Divide file into n_workers byte ranges
    chunk_sz = total // n_workers
    ranges = [
        (i * chunk_sz, (i + 1) * chunk_sz - 1 if i < n_workers - 1 else total - 1)
        for i in range(n_workers)
    ]

    tmp_dir = dest + ".parts"
    os.makedirs(tmp_dir, exist_ok=True)
    parts = [os.path.join(tmp_dir, f"p{i:04d}") for i in range(n_workers)]

    try:
        success = True
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(_dl_chunk, session, url, s, e, parts[i], tracker): i
                for i, (s, e) in enumerate(ranges)
            }
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as ex:
                    log(f"Chunk {futs[fut]} failed: {ex}")
                    success = False

        if not success:
            return False

        # Reassemble parts in order
        with open(dest, "wb") as out:
            for p in parts:
                with open(p, "rb") as inp:
                    shutil.copyfileobj(inp, out)
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── text extraction ───────────────────────────────────────────────────────────

TEXT_FIELD_CANDIDATES = [
    "text", "content", "code", "body", "document", "passage",
    "response", "answer", "instruction", "output", "messages",
    "conversations", "question", "abstract",
]


def extract_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role") or item.get("from") or ""
                body = (
                    item.get("content") or item.get("value")
                    or item.get("body") or item.get("text") or ""
                )
                if isinstance(body, str) and body:
                    parts.append(f"{role}: {body}" if role else body)
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(filter(None, parts))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def pick_text_field(columns: list[str], forced: str | None) -> str | None:
    if forced and forced in columns:
        return forced
    for c in TEXT_FIELD_CANDIDATES:
        if c in columns:
            return c
    return columns[0] if columns else None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset to JSONL using parallel chunk downloading."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-gb", type=float, default=0.0,
                        help="Maximum GB to write (0 = unlimited)")
    parser.add_argument("--subset", default=None,
                        help="Dataset config/subset name (e.g. '20231101.en')")
    parser.add_argument("--split", default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace API token for gated datasets")
    parser.add_argument("--text-field", default=None,
                        help="Override text field name")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel HTTP connections per file (default: 10)")
    args = parser.parse_args()

    try:
        import requests
        from huggingface_hub import list_repo_files, hf_hub_url  # type: ignore
        import pyarrow.parquet as pq                              # type: ignore
    except ImportError as e:
        print(f"ERROR:Missing package: {e}", flush=True)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    safe_id = args.dataset_id.replace("/", "__")
    out_path = os.path.join(args.output_dir, f"{safe_id}.jsonl")
    max_bytes = int(args.max_gb * 1024 ** 3) if args.max_gb > 0 else 0

    # Build a requests session with optional auth
    session = requests.Session()
    if args.hf_token:
        session.headers["Authorization"] = f"Bearer {args.hf_token}"
    session.headers["User-Agent"] = "quark-dataset-downloader/2.0"

    log(f"Dataset : {args.dataset_id}")
    if args.subset:
        log(f"  subset : {args.subset}")
    log(f"  split  : {args.split}")
    log(f"  output : {out_path}")
    log(f"  workers: {args.workers} parallel connections per file")
    if max_bytes:
        log(f"  limit  : {args.max_gb:.1f} GB")

    # ── enumerate Parquet shards ───────────────────────────────────────────
    log("Listing dataset files on HuggingFace Hub…")
    try:
        all_files = sorted(list_repo_files(
            args.dataset_id,
            repo_type="dataset",
            token=args.hf_token,
        ))
    except Exception as e:
        log(f"Cannot list files ({e}) — falling back to streaming mode")
        _streaming_fallback(args, out_path, max_bytes)
        return

    split = args.split
    subset = args.subset

    def matches(path: str) -> bool:
        if not path.endswith(".parquet"):
            return False
        if split and split not in path:
            return False
        if subset and subset not in path:
            return False
        return True

    parquet_files = [f for f in all_files if matches(f)]

    # Progressive relaxation — drop subset filter, then split filter
    if not parquet_files:
        parquet_files = [f for f in all_files if f.endswith(".parquet") and split in f]
    if not parquet_files:
        parquet_files = [f for f in all_files if f.endswith(".parquet")]

    if not parquet_files:
        log("No Parquet shards found — falling back to streaming mode")
        _streaming_fallback(args, out_path, max_bytes)
        return

    log(f"Found {len(parquet_files)} Parquet shards")

    # ── estimate total download size (sample first 3 shards via HEAD) ─────
    def get_size(path: str) -> int:
        url = hf_hub_url(args.dataset_id, path, repo_type="dataset")
        try:
            r = session.head(url, timeout=20, allow_redirects=True)
            return int(r.headers.get("Content-Length", 0))
        except Exception:
            return 0

    sample_n = min(3, len(parquet_files))
    sample_sizes = [get_size(f) for f in parquet_files[:sample_n]]
    avg_shard = sum(sample_sizes) / max(sample_n, 1)

    # Trim shard list to honour the GB cap
    if max_bytes and avg_shard > 0:
        n_cap = max(1, int(max_bytes / avg_shard))
        if n_cap < len(parquet_files):
            parquet_files = parquet_files[:n_cap]
            log(f"Cap {args.max_gb:.1f} GB → keeping first {n_cap} shards")

    estimated_total = int(avg_shard * len(parquet_files)) if avg_shard > 0 else 0
    log(
        f"Estimated download: "
        f"{estimated_total / 1024**3:.2f} GB "
        f"({len(parquet_files)} shards × ~{avg_shard / 1024**2:.0f} MiB each)"
    )

    tracker = SpeedTracker()
    total_ref: list[int] = [estimated_total]
    stop_event = threading.Event()
    start_reporter(tracker, lambda: total_ref[0], stop_event)

    written_bytes = 0

    with tempfile.TemporaryDirectory(prefix="quark_hf_") as tmp_dir:
        with open(out_path, "a", encoding="utf-8") as fout:
            text_field: str | None = None

            for idx, shard_path in enumerate(parquet_files):
                url = hf_hub_url(args.dataset_id, shard_path, repo_type="dataset")
                name = os.path.basename(shard_path)
                emit_file(name)
                log(
                    f"━━  Shard {idx + 1}/{len(parquet_files)}: {name}  "
                    f"({idx * 100 // len(parquet_files)}% of shards done)"
                )

                local = os.path.join(tmp_dir, name)
                ok = parallel_download(session, url, local, args.workers, tracker)
                if not ok:
                    log(f"⚠ Skipping {name} (download failed)")
                    continue

                # Convert Parquet → JSONL rows
                try:
                    table = pq.read_table(local)
                    cols = table.schema.names
                    if text_field is None:
                        text_field = pick_text_field(cols, args.text_field)
                        if text_field is None:
                            log(f"Cannot find text field in columns: {cols}")
                            os.remove(local)
                            continue
                        log(f"Text field detected: '{text_field}'")

                    col_data = table.column(text_field).to_pylist()
                    rows = 0
                    for value in col_data:
                        txt = extract_text(value)
                        if not txt.strip():
                            continue
                        line = json.dumps({"text": txt}, ensure_ascii=False) + "\n"
                        fout.write(line)
                        written_bytes += len(line.encode("utf-8"))
                        rows += 1
                        if max_bytes and written_bytes >= max_bytes:
                            break
                    log(f"  {rows:,} rows → {written_bytes / 1024**2:.1f} MiB written so far")
                    os.remove(local)
                except Exception as e:
                    log(f"⚠ Parquet read error on {name}: {e}")
                    continue

                if max_bytes and written_bytes >= max_bytes:
                    log(f"Reached {args.max_gb:.1f} GB limit — stopping.")
                    break

    stop_event.set()
    mb = written_bytes / 1024 ** 2
    log(f"✅  Finished: {mb:.1f} MiB written → {out_path}")
    emit_progress(1.0)
    emit_speed(0.0)
    print("DONE", flush=True)


# ── streaming fallback ────────────────────────────────────────────────────────

def _streaming_fallback(args, out_path: str, max_bytes: int) -> None:
    """Row-by-row HuggingFace streaming — used when Parquet files are not
    directly accessible (e.g. trust_remote_code datasets with custom loaders)."""
    log("Streaming mode — row-by-row, no parallel acceleration")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR:datasets package not installed", flush=True)
        sys.exit(1)

    kw: dict = dict(streaming=True, trust_remote_code=True)
    if args.hf_token:
        kw["token"] = args.hf_token
    if args.subset:
        kw["name"] = args.subset

    try:
        ds = load_dataset(args.dataset_id, split=args.split, **kw)
    except Exception as e:
        print(f"ERROR:{e}", flush=True)
        sys.exit(1)

    written_bytes = 0
    written_rows = 0
    text_field: str | None = None

    with open(out_path, "a", encoding="utf-8") as fout:
        for row in ds:
            if text_field is None:
                candidates = (
                    ([args.text_field] if args.text_field else []) + TEXT_FIELD_CANDIDATES
                )
                text_field = next((c for c in candidates if c and c in row), None)
                if text_field is None:
                    text_field = next((k for k, v in row.items() if v is not None), None)
                if text_field is None:
                    print("ERROR:Cannot determine text field.", flush=True)
                    sys.exit(1)
                log(f"Text field: '{text_field}'")

            txt = extract_text(row.get(text_field, ""))
            if not txt.strip():
                continue

            line = json.dumps({"text": txt}, ensure_ascii=False) + "\n"
            fout.write(line)
            written_bytes += len(line.encode("utf-8"))
            written_rows += 1

            if written_rows % 500 == 0:
                mb = written_bytes / 1024 ** 2
                log(f"Wrote {written_rows:,} rows ({mb:.1f} MiB)")
                if max_bytes:
                    emit_progress(min(written_bytes / max_bytes, 0.99))

            if max_bytes and written_bytes >= max_bytes:
                log(f"Reached {args.max_gb:.1f} GB limit — stopping.")
                break

    mb = written_bytes / 1024 ** 2
    log(f"✅  Finished: {written_rows:,} rows, {mb:.1f} MiB → {out_path}")
    emit_progress(1.0)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
