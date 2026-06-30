"""Jamendo dataset utilities for retraining and evaluation.

This module provides a small, reproducible pipeline for:
- querying the Jamendo API,
- filtering usable tracks,
- downloading audio files,
- writing a metadata manifest, and
- creating train/validation/test splits by track id.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

DEFAULT_API_URL = "https://api.jamendo.com/v3.0/tracks/"
DEFAULT_AUDIO_FORMAT = "mp32"
DEFAULT_INCLUDE = "musicinfo"
DEFAULT_LIMIT = 200
DEFAULT_TIMEOUT = 20
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class JamendoConfig:
    client_id: str
    api_url: str = DEFAULT_API_URL
    audioformat: str = DEFAULT_AUDIO_FORMAT
    include: str = DEFAULT_INCLUDE
    limit: int = DEFAULT_LIMIT
    timeout: int = DEFAULT_TIMEOUT


class JamendoError(RuntimeError):
    """Raised when Jamendo requests or validation fail."""


def _resolve_download_url(entry: Dict[str, Any]) -> str:
    for key in ("audiodownload", "audio", "audiodownloadpreview", "zipdownload"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_genre(entry: Dict[str, Any]) -> str:
    musicinfo = entry.get("musicinfo") or {}
    tags = musicinfo.get("tags") or {}
    genres = tags.get("genres") or []
    if isinstance(genres, list) and genres:
        first = genres[0]
        if isinstance(first, dict):
            return str(first.get("name", "")).strip()
        return str(first).strip()
    return ""


def normalize_track_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    track_id = str(entry.get("id", "")).strip()
    if not track_id:
        return None

    genre = _extract_genre(entry)
    if not genre:
        return None

    download_url = _resolve_download_url(entry)
    if not download_url:
        return None

    duration = entry.get("duration", "")
    try:
        duration_sec = float(duration) if duration not in (None, "") else ""
    except (TypeError, ValueError):
        duration_sec = ""

    return {
        "track_id": track_id,
        "title": str(entry.get("name", "")).strip(),
        "artist": str(entry.get("artist_name", "")).strip(),
        "album": str(entry.get("album_name", "")).strip(),
        "duration_sec": duration_sec,
        "download_url": download_url,
        "filename": f"{track_id}.mp3",
        "genre": genre,
    }


def fetch_tracks_page(config: JamendoConfig, offset: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    page_limit = limit or config.limit
    response = requests.get(
        config.api_url,
        params={
            "client_id": config.client_id,
            "format": "json",
            "include": config.include,
            "audioformat": config.audioformat,
            "limit": page_limit,
            "offset": offset,
        },
        timeout=config.timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("results", []) or []


def collect_tracks(
    config: JamendoConfig,
    max_tracks: int,
    require_genre: bool = True,
    limit_per_page: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not config.client_id:
        raise JamendoError("A Jamendo client id is required. Set JAMENDO_CLIENT_ID before running.")

    collected: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    offset = 0
    page_limit = limit_per_page or config.limit

    while len(collected) < max_tracks:
        results = fetch_tracks_page(config, offset=offset, limit=page_limit)
        if not results:
            break

        for entry in results:
            normalized = normalize_track_entry(entry)
            if normalized is None:
                continue
            if require_genre and not normalized["genre"]:
                continue
            if normalized["track_id"] in seen_ids:
                continue
            seen_ids.add(normalized["track_id"])
            collected.append(normalized)
            if len(collected) >= max_tracks:
                break

        offset += page_limit

    return collected


def download_file(url: str, destination: Path, timeout: int = DEFAULT_TIMEOUT) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            tmp_path = destination.with_suffix(destination.suffix + ".part")
            with open(tmp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        handle.write(chunk)
            tmp_path.replace(destination)
        return True
    except Exception:
        if destination.with_suffix(destination.suffix + ".part").exists():
            destination.with_suffix(destination.suffix + ".part").unlink(missing_ok=True)
        return False


def write_manifest(rows: Sequence[Dict[str, Any]], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["track_id", "title", "artist", "album", "duration_sec", "download_url", "filename", "genre"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def download_dataset(
    client_id: str,
    output_dir: str | Path,
    max_tracks: int = 1000,
    require_genre: bool = True,
    download_audio: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Path:
    output_path = Path(output_dir)
    audio_dir = output_path / "downloaded_tracks"
    audio_dir.mkdir(parents=True, exist_ok=True)

    config = JamendoConfig(client_id=client_id)
    tracks = collect_tracks(config, max_tracks=max_tracks, require_genre=require_genre)
    manifest_path = write_manifest(tracks, output_path / "metadata.csv")

    failures: List[str] = []
    if download_audio:
        for row in tracks:
            destination = audio_dir / row["filename"]
            if destination.exists() and destination.stat().st_size > 0:
                continue
            ok = download_file(row["download_url"], destination, timeout=config.timeout)
            if not ok:
                failures.append(row["track_id"])

    if failures:
        with open(output_path / "failed_ids.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(failures))

    return manifest_path


def split_manifest(
    manifest_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Path]:
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    manifest_path = Path(manifest_csv)
    output_path = Path(output_dir)
    df = pd.read_csv(manifest_path)
    unique_ids = df["track_id"].astype(str).drop_duplicates().tolist()

    train_ids, temp_ids = train_test_split(unique_ids, test_size=(1.0 - train_ratio), random_state=seed)
    if val_ratio == 0.0 or test_ratio == 0.0:
        val_ids = temp_ids
        test_ids = []
    else:
        relative_test = test_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(temp_ids, test_size=relative_test, random_state=seed)

    splits = {
        "train": df[df["track_id"].astype(str).isin(train_ids)].copy(),
        "val": df[df["track_id"].astype(str).isin(val_ids)].copy(),
        "test": df[df["track_id"].astype(str).isin(test_ids)].copy(),
    }

    output_path.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}
    for split_name, split_df in splits.items():
        split_path = output_path / f"{split_name}_metadata.csv"
        split_df.to_csv(split_path, index=False)
        results[split_name] = split_path

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and split a Jamendo dataset")
    parser.add_argument("--client-id", default=os.environ.get("JAMENDO_CLIENT_ID", ""), help="Jamendo client id")
    parser.add_argument("--output-dir", default="jamendo_dataset", help="Directory for metadata and audio")
    parser.add_argument("--max-tracks", type=int, default=1000, help="Maximum number of tracks to collect")
    parser.add_argument("--skip-audio", action="store_true", help="Only write metadata, do not download audio")
    parser.add_argument("--no-genre-filter", action="store_true", help="Keep tracks even if genre is missing")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.client_id:
        raise JamendoError("Missing Jamendo client id. Pass --client-id or set JAMENDO_CLIENT_ID.")

    output_dir = Path(args.output_dir)
    manifest_path = download_dataset(
        client_id=args.client_id,
        output_dir=output_dir,
        max_tracks=args.max_tracks,
        require_genre=not args.no_genre_filter,
        download_audio=not args.skip_audio,
        random_state=args.seed,
    )
    split_manifest(
        manifest_csv=manifest_path,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Wrote manifest to {manifest_path}")
    print(f"Created split CSVs in {output_dir}")


if __name__ == "__main__":
    main()
