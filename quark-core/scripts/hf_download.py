#!/usr/bin/env python3
"""Quark HuggingFace dataset downloader.

Streams rows from a HuggingFace dataset into a JSONL file.

Protocol on stdout:
  LOG:<message>       — human-readable status line
  PROGRESS:<0..1>     — download progress (0.0 – 1.0)
  DONE                — completed successfully
  ERROR:<message>     — fatal error (also causes non-zero exit)
"""

import argparse
import json
import os
import sys


def log(msg: str) -> None:
    print(f"LOG:{msg}", flush=True)


def progress(p: float) -> None:
    print(f"PROGRESS:{p:.4f}", flush=True)


def extract_text(value) -> str:
    """Convert any dataset field value to a plain text string."""
    if isinstance(value, str):
        return value
    elif isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role") or item.get("from") or ""
                body = (
                    item.get("content")
                    or item.get("value")
                    or item.get("body")
                    or item.get("text")
                    or ""
                )
                if isinstance(body, str) and body:
                    parts.append(f"{role}: {body}" if role else body)
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(filter(None, parts))
    elif isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    elif value is None:
        return ""
    else:
        return str(value)


TEXT_FIELD_CANDIDATES = [
    "text",
    "content",
    "code",
    "body",
    "document",
    "passage",
    "response",
    "answer",
    "instruction",
    "output",
    "messages",
    "conversations",
    "question",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a HuggingFace dataset to JSONL.")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="HuggingFace dataset ID (e.g. 'codeparrot/github-code-clean')",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write JSONL file(s) into",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=0.0,
        help="Max gigabytes to write per file (0 = unlimited)",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Dataset configuration/subset name (e.g. '20231101.en' for Wikipedia)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream (default: train)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace API token (required for gated datasets such as The Stack)",
    )
    parser.add_argument(
        "--text-field",
        default=None,
        help="Override the field name to extract as text",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        print("ERROR:The 'datasets' package is not installed.", flush=True)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    safe_id = args.dataset_id.replace("/", "__")
    out_path = os.path.join(args.output_dir, f"{safe_id}.jsonl")
    max_bytes = int(args.max_gb * 1024 * 1024 * 1024) if args.max_gb > 0 else 0

    log(f"Dataset : {args.dataset_id}")
    if args.subset:
        log(f"  subset : {args.subset}")
    log(f"  split  : {args.split}")
    log(f"  output : {out_path}")
    if max_bytes:
        log(f"  limit  : {args.max_gb:.1f} GB")

    load_kwargs: dict = dict(
        streaming=True,
        trust_remote_code=True,
    )
    if args.hf_token:
        load_kwargs["token"] = args.hf_token
    if args.subset:
        load_kwargs["name"] = args.subset

    try:
        ds = load_dataset(args.dataset_id, split=args.split, **load_kwargs)
    except Exception as exc:
        print(f"ERROR:{exc}", flush=True)
        sys.exit(1)

    forced_field: str | None = args.text_field
    detected_field: str | None = None
    written_bytes = 0
    written_rows = 0
    REPORT_EVERY = 500

    try:
        with open(out_path, "a", encoding="utf-8") as fout:
            for row in ds:
                # Detect text field on first non-empty row
                if detected_field is None:
                    candidates = ([forced_field] if forced_field else []) + TEXT_FIELD_CANDIDATES
                    for candidate in candidates:
                        if candidate and candidate in row:
                            detected_field = candidate
                            break
                    if detected_field is None:
                        # Fall back to the first non-None field
                        for k, v in row.items():
                            if v is not None:
                                detected_field = k
                                break
                    if detected_field is None:
                        print(
                            "ERROR:Cannot determine text field. Use --text-field.",
                            flush=True,
                        )
                        sys.exit(1)
                    log(f"Text field: '{detected_field}'")

                raw = row.get(detected_field, "")
                text = extract_text(raw)
                if not text.strip():
                    continue

                line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                encoded = line.encode("utf-8")
                fout.write(line)
                written_bytes += len(encoded)
                written_rows += 1

                if written_rows % REPORT_EVERY == 0:
                    mb = written_bytes / (1024 * 1024)
                    log(f"Wrote {written_rows:,} rows  ({mb:.1f} MB)")
                    if max_bytes:
                        progress(min(written_bytes / max_bytes, 0.99))

                if max_bytes and written_bytes >= max_bytes:
                    log(f"Reached {args.max_gb:.1f} GB limit — stopping.")
                    break

    except KeyboardInterrupt:
        print("ERROR:Interrupted by user.", flush=True)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR:{exc}", flush=True)
        sys.exit(1)

    mb = written_bytes / (1024 * 1024)
    log(f"✅  Finished: {written_rows:,} rows, {mb:.1f} MB → {out_path}")
    progress(1.0)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
