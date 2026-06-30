import os
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATHS = [
    Path(os.environ.get("MODEL_PATH", "")) if os.environ.get("MODEL_PATH") else None,
    Path(__file__).resolve().parent / "embdmodel.keras",
    PROJECT_ROOT / "embdmodel.keras",
]
DEFAULT_REFERENCE_INDEX_PATHS = [
    Path(os.environ.get("REFERENCE_INDEX_PATH", "")) if os.environ.get("REFERENCE_INDEX_PATH") else None,
    Path(__file__).resolve().parent / "dict.pickle",
    PROJECT_ROOT / "dict.pickle",
]
MODEL_PATH = next((path for path in DEFAULT_MODEL_PATHS if path and path.exists()), None)

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find embdmodel.keras. Place it in backend/embdmodel.keras or the project root, or set MODEL_PATH."
    )


def load_embedding_model() -> tf.keras.Model:
    model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
    try:
        return model.layers[2]
    except (AttributeError, IndexError) as exc:
        raise RuntimeError("Loaded model does not expose an encoder at layers[2].") from exc


embedding_model = load_embedding_model()

app = FastAPI(title="Music Recognition API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecognizeResponse(BaseModel):
    matched_song: str | None
    matched_distance: float | None
    candidates: list[dict[str, Any]]
    embedding_shape: list[int]


def load_audio_to_embedding(audio_bytes: bytes, filename: str | None = None) -> np.ndarray:
    suffix = Path(filename).suffix if filename else ""
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()

        signal, sample_rate = librosa.load(temp_audio.name, sr=None, mono=True)
        if signal.size == 0:
            raise ValueError("Uploaded audio file is empty or unreadable.")

        duration = librosa.get_duration(y=signal, sr=sample_rate)
        if duration > 10:
            signal = signal[: int(sample_rate * 10)]

        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
        spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        spec_min = float(np.min(spec_db))
        spec_max = float(np.max(spec_db))
        if spec_max > spec_min:
            spec_db = (spec_db - spec_min) / (spec_max - spec_min)
        else:
            spec_db = np.zeros_like(spec_db)

        spec_rgb = np.stack([spec_db, spec_db, spec_db], axis=-1)
        spec_rgb = tf.image.resize(spec_rgb, (150, 150)).numpy().astype(np.float32)
        return np.expand_dims(spec_rgb, axis=0)


def signal_to_embedding_input(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    if signal.size == 0:
        raise ValueError("Audio signal is empty or unreadable.")

    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
    spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    spec_min = float(np.min(spec_db))
    spec_max = float(np.max(spec_db))
    if spec_max > spec_min:
        spec_db = (spec_db - spec_min) / (spec_max - spec_min)
    else:
        spec_db = np.zeros_like(spec_db)

    spec_rgb = np.stack([spec_db, spec_db, spec_db], axis=-1)
    spec_rgb = tf.image.resize(spec_rgb, (150, 150)).numpy().astype(np.float32)
    return np.expand_dims(spec_rgb, axis=0)


def predict_embedding(audio_bytes: bytes, filename: str | None = None) -> np.ndarray:
    image = load_audio_to_embedding(audio_bytes, filename=filename)
    embedding = embedding_model.predict(image, verbose=0)
    return np.asarray(embedding[0], dtype=np.float32)


def load_reference_index() -> dict[str, list[np.ndarray]]:
    path = next((candidate for candidate in DEFAULT_REFERENCE_INDEX_PATHS if candidate and candidate.exists()), None)
    if path is None:
        return {}

    import pickle

    with path.open("rb") as handle:
        raw_index = pickle.load(handle)

    converted: dict[str, list[np.ndarray]] = {}
    for key, values in raw_index.items():
        converted[key] = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    return converted


REFERENCE_INDEX = load_reference_index()


def rank_matches(query_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for song_name, embeddings in REFERENCE_INDEX.items():
        if not embeddings:
            continue
        distances = [float(np.linalg.norm(query_embedding - embedding, ord=1)) for embedding in embeddings]
        scored.append({"song": song_name, "distance": min(distances)})

    scored.sort(key=lambda item: item["distance"])
    return scored[:top_k]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "embedding_shape": list(embedding_model.output_shape),
        "reference_songs": len(REFERENCE_INDEX),
    }


@app.post("/embed")
async def embed_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data received.")

    try:
        embedding = predict_embedding(audio_bytes, filename=file.filename)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not process audio file: {exc}") from exc

    return {
        "filename": file.filename,
        "embedding": embedding.tolist(),
        "embedding_shape": list(embedding.shape),
    }


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize_audio(file: UploadFile = File(...), top_k: int = 5) -> RecognizeResponse:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data received.")

    if not REFERENCE_INDEX:
        raise HTTPException(
            status_code=503,
            detail=(
                "No reference index is loaded. Build a dict.pickle from the Jamendo tracks "
                "or set REFERENCE_INDEX_PATH to an existing pickle file before calling /recognize."
            ),
        )

    try:
        query_embedding = predict_embedding(audio_bytes, filename=file.filename)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not process audio file: {exc}") from exc

    candidates = rank_matches(query_embedding, top_k=top_k)
    best_match = candidates[0] if candidates else None

    return RecognizeResponse(
        matched_song=best_match["song"] if best_match else None,
        matched_distance=best_match["distance"] if best_match else None,
        candidates=candidates,
        embedding_shape=list(query_embedding.shape),
    )
