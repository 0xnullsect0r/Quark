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
  PAUSED:<reason>         — paused cleanly; progress saved to disk
  DONE                    — all work complete
  ERROR:<message>         — fatal error (also causes non-zero exit)

Crash recovery / pause:
  A progress file is maintained at <out_path>.progress.json, tracking which
  Parquet shards have been fully converted and the output file byte offset after
  each completed shard.  On resume, the output file is truncated to the last
  good offset (discarding any partial shard write from a crash) and completed
  shards are skipped.

  If --stop-file is given, the script checks for that file at the start of each
  shard.  When found it deletes the file, emits PAUSED:, and exits 0 so the
  caller can resume later.
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


# ── progress file helpers ─────────────────────────────────────────────────────

def load_progress(progress_file: str) -> tuple[set[str], int]:
    """Return (completed_shards, output_size_bytes) from the progress file."""
    try:
        with open(progress_file, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("completed_shards", [])), int(data.get("output_size_bytes", 0))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return set(), 0


def save_progress(progress_file: str, completed_shards: set[str], output_size_bytes: int) -> None:
    tmp = progress_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"completed_shards": sorted(completed_shards), "output_size_bytes": output_size_bytes},
            f,
        )
    os.replace(tmp, progress_file)  # atomic on POSIX; best-effort on Windows


# ── pause signal check ────────────────────────────────────────────────────────

def check_pause(stop_file: str | None) -> bool:
    """Return True and delete the stop file if a pause has been requested."""
    if stop_file and os.path.exists(stop_file):
        try:
            os.remove(stop_file)
        except OSError:
            pass
        return True
    return False


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
    parser.add_argument("--stop-file", default=None,
                        help="Path to pause-signal file; when it appears, checkpoint and exit 0")
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
    progress_file = out_path + ".progress.json"
    max_bytes = int(args.max_gb * 1024 ** 3) if args.max_gb > 0 else 0

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

    # ── load prior progress and truncate output file to last safe offset ──
    completed_shards, saved_output_size = load_progress(progress_file)
    if completed_shards:
        log(f"ℹ  Resuming: {len(completed_shards)} shard(s) already complete")
        if os.path.exists(out_path) and saved_output_size > 0:
            current_size = os.path.getsize(out_path)
            if current_size != saved_output_size:
                log(f"ℹ  Truncating output from {current_size} → {saved_output_size} bytes "
                    f"(discarding partial shard from last crash)")
                with open(out_path, "r+b") as ftrunc:
                    ftrunc.truncate(saved_output_size)

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
        _streaming_fallback(args, out_path, progress_file, completed_shards, max_bytes)
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

    if not parquet_files:
        parquet_files = [f for f in all_files if f.endswith(".parquet") and split in f]
    if not parquet_files:
        parquet_files = [f for f in all_files if f.endswith(".parquet")]

    if not parquet_files:
        log("No Parquet shards found — falling back to streaming mode")
        _streaming_fallback(args, out_path, progress_file, completed_shards, max_bytes)
        return

    remaining = [f for f in parquet_files if f not in completed_shards]
    log(f"Found {len(parquet_files)} Parquet shard(s)  "
        f"({len(completed_shards)} already done, {len(remaining)} to download)")

    # ── estimate total download size ──────────────────────────────────────
    def get_size(path: str) -> int:
        url = hf_hub_url(args.dataset_id, path, repo_type="dataset")
        try:
            r = session.head(url, timeout=20, allow_redirects=True)
            return int(r.headers.get("Content-Length", 0))
        except Exception:
            return 0

    sample_n = min(3, len(remaining)) if remaining else min(3, len(parquet_files))
    sample_pool = remaining if remaining else parquet_files
    sample_sizes = [get_size(f) for f in sample_pool[:sample_n]]
    avg_shard = sum(sample_sizes) / max(sample_n, 1)

    if max_bytes and avg_shard > 0:
        n_cap = max(1, int(max_bytes / avg_shard))
        if n_cap < len(parquet_files):
            parquet_files = parquet_files[:n_cap]
            remaining = [f for f in parquet_files if f not in completed_shards]
            log(f"Cap {args.max_gb:.1f} GB → keeping first {n_cap} shards")

    estimated_total = int(avg_shard * len(remaining)) if avg_shard > 0 else 0
    log(
        f"Estimated remaining download: "
        f"{estimated_total / 1024**3:.2f} GB "
        f"({len(remaining)} shards × ~{avg_shard / 1024**2:.0f} MiB each)"
    )

    tracker = SpeedTracker()
    total_ref: list[int] = [estimated_total]
    stop_event = threading.Event()
    start_reporter(tracker, lambda: total_ref[0], stop_event)

    written_bytes = saved_output_size  # bytes already on disk from prior runs

    with tempfile.TemporaryDirectory(prefix="quark_hf_") as tmp_dir:
        with open(out_path, "a", encoding="utf-8") as fout:
            text_field: str | None = None

            for idx, shard_path in enumerate(parquet_files):
                # ── pause check (before starting this shard) ──────────────
                if check_pause(args.stop_file):
                    stop_event.set()
                    fout.flush()
                    size_now = fout.seek(0, 2)
                    save_progress(progress_file, completed_shards, size_now)
                    log(f"⏸  Paused after {len(completed_shards)} shard(s) — progress saved.")
                    print(f"PAUSED:Paused after {len(completed_shards)} shard(s) — "
                          f"click Resume to continue.", flush=True)
                    return

                # ── skip already-completed shards ─────────────────────────
                if shard_path in completed_shards:
                    log(f"↩  Skipping already-completed shard: {os.path.basename(shard_path)}")
                    continue

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

                # ── convert Parquet → JSONL ────────────────────────────────
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

                # ── checkpoint: flush + record this shard as complete ──────
                fout.flush()
                try:
                    os.fsync(fout.fileno())
                except OSError:
                    pass
                size_now = fout.seek(0, 2)  # seek to end → current byte offset
                completed_shards.add(shard_path)
                save_progress(progress_file, completed_shards, size_now)
                log(f"  ✔ Checkpoint saved ({len(completed_shards)} shard(s) done, "
                    f"{size_now / 1024**2:.1f} MiB on disk)")

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

def _streaming_fallback(args, out_path: str, progress_file: str,
                        completed_shards: set[str], max_bytes: int) -> None:
    """Row-by-row HuggingFace streaming — used when Parquet files are not
    directly accessible.  Supports pause via stop file (checked every 500 rows).
    Progress is saved as a synthetic shard key 'stream:<rows_written>'."""
    log("Streaming mode — row-by-row, no parallel acceleration")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR:datasets package not installed", flush=True)
        sys.exit(1)

    # Recover the row count written so far from a previously saved stream key.
    rows_already = 0
    for key in completed_shards:
        if key.startswith("stream:"):
            try:
                rows_already = int(key.split(":", 1)[1])
            except ValueError:
                pass

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
            # Skip rows already written in a prior run.
            if written_rows < rows_already:
                written_rows += 1
                continue

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

                # Checkpoint every 500 rows in streaming mode.
                fout.flush()
                try:
                    os.fsync(fout.fileno())
                except OSError:
                    pass
                size_now = fout.seek(0, 2)
                stream_key = f"stream:{written_rows}"
                # Keep only the latest stream key to avoid unbounded growth.
                stale = {k for k in completed_shards if k.startswith("stream:")}
                completed_shards -= stale
                completed_shards.add(stream_key)
                save_progress(progress_file, completed_shards, size_now)

                # Pause check every 500 rows.
                if check_pause(args.stop_file):
                    fout.flush()
                    size_now = fout.seek(0, 2)
                    save_progress(progress_file, completed_shards, size_now)
                    log(f"⏸  Paused at row {written_rows} — progress saved.")
                    print(f"PAUSED:Paused at row {written_rows} — "
                          f"click Resume to continue.", flush=True)
                    return

            if max_bytes and written_bytes >= max_bytes:
                log(f"Reached {args.max_gb:.1f} GB limit — stopping.")
                break

    mb = written_bytes / 1024 ** 2
    log(f"✅  Finished: {written_rows:,} rows, {mb:.1f} MiB → {out_path}")
    emit_progress(1.0)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
