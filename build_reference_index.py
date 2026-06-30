"""Build a reference embedding index from Jamendo audio files.

The generated pickle matches the structure expected by backend.app:
{song_name: [embedding_1, embedding_2, ...]}
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import librosa
import numpy as np
import pandas as pd

from backend.app import embedding_model, signal_to_embedding_input


def iter_segments(signal: np.ndarray, sample_rate: int, segment_seconds: float = 10.0) -> Iterable[np.ndarray]:
    samples_per_segment = max(1, int(sample_rate * segment_seconds))
    for start in range(0, signal.size, samples_per_segment):
        segment = signal[start : start + samples_per_segment]
        if segment.size:
            yield segment


def load_metadata(metadata_csv: Path) -> Dict[str, str]:
    if not metadata_csv.exists():
        return {}

    frame = pd.read_csv(metadata_csv)
    mapping: Dict[str, str] = {}
    for row in frame.itertuples(index=False):
        track_id = str(getattr(row, "track_id", "")).strip()
        title = str(getattr(row, "title", "")).strip()
        artist = str(getattr(row, "artist", "")).strip()
        if track_id:
            mapping[track_id] = f"{artist} - {title}.mp3" if artist or title else f"{track_id}.mp3"
    return mapping


def build_reference_index(audio_dir: Path, metadata_csv: Path, output_path: Path) -> Path:
    metadata_labels = load_metadata(metadata_csv)
    reference_index: Dict[str, List[np.ndarray]] = {}

    audio_files = sorted(audio_dir.glob("*.mp3"))
    if not audio_files:
        raise FileNotFoundError(f"No .mp3 files found in {audio_dir}")

    for audio_path in audio_files:
        track_id = audio_path.stem
        song_name = metadata_labels.get(track_id, audio_path.name)

        signal, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        embeddings: List[np.ndarray] = []
        for segment in iter_segments(signal, sample_rate):
            model_input = signal_to_embedding_input(segment, sample_rate)
            embedding = embedding_model.predict(model_input, verbose=0)[0].astype(np.float32)
            embeddings.append(embedding)

        if embeddings:
            reference_index[song_name] = embeddings

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(reference_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a song embedding reference index")
    parser.add_argument("--audio-dir", default="jamendo_dataset/downloaded_tracks", help="Directory with MP3 files")
    parser.add_argument("--metadata-csv", default="jamendo_dataset/metadata.csv", help="Jamendo metadata CSV")
    parser.add_argument("--output", default="dict.pickle", help="Output pickle path")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_path = build_reference_index(Path(args.audio_dir), Path(args.metadata_csv), Path(args.output))
    print(f"Wrote reference index to {output_path}")


if __name__ == "__main__":
    main()