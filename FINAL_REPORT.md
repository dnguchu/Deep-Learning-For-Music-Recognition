# Final Report: Deep Learning for Music Recognition

## Project Overview

This project implements a music recognition system built around a **Siamese convolutional neural network**. The system learns a compact audio embedding from mel-spectrogram images, then compares query embeddings against a cached reference index to identify the most likely matching song. The codebase includes the full training workflow, a Jamendo data preparation pipeline, a reference index builder, and a serving stack with FastAPI and Streamlit.

The implementation is centered on three layers:

1. **Training and representation learning** in [training.ipynb](training.ipynb)
2. **Reference embedding generation and matching** in [test.ipynb](test.ipynb) and [build_reference_index.py](build_reference_index.py)
3. **Deployment and user interaction** in [backend/app.py](backend/app.py) and [frontend/streamlit_app.py](frontend/streamlit_app.py)

---

## 1. Problem Being Solved

The goal is not to classify audio into a fixed set of labels directly. Instead, the model learns whether two audio segments come from the same song. That choice makes the system more flexible because new songs can be added by computing embeddings and storing them in a reference dictionary, without retraining the entire network.

This is a good fit for music recognition because:

- songs can be split into short, comparable segments
- mel-spectrograms preserve time-frequency structure in a CNN-friendly format
- similarity learning generalizes better than a rigid single-label classifier
- inference can be made efficient through cached embeddings

---

## 2. Dataset and Data Preparation

### Jamendo pipeline

The repository includes [jamendo_pipeline.py](jamendo_pipeline.py), which provides a reproducible way to collect and organize Jamendo tracks. Its main responsibilities are:

- querying the Jamendo API for track metadata
- filtering unusable tracks
- downloading audio files into a local `downloaded_tracks` folder
- writing a clean metadata manifest to CSV
- creating train, validation, and test splits by track id

The pipeline keeps the split at the **track level**, which helps prevent leakage between training and evaluation segments from the same song.

### Generated dataset artifacts

The implementation works with the following local dataset layout:

- `jamendo_dataset/metadata.csv`
- `jamendo_dataset/downloaded_tracks/`
- optional split CSVs created by the Jamendo utility

The reference documentation also notes that the split is produced inside the notebook workflow rather than stored as a separate static file in the repository.

### Audio segmentation

In the training and reference-index workflows, songs are divided into **10-second segments**. That segmentation is important because it gives the model consistent, local audio windows instead of forcing it to learn from full-length tracks of variable size.

---

## 3. Preprocessing Pipeline

The preprocessing pipeline converts raw audio into a normalized image representation that the CNN can consume.

### Audio to spectrogram

For each audio segment, the implementation performs the following steps:

1. Load the audio with `librosa`
2. Convert the waveform into a mel-spectrogram
3. Convert power values to decibels with `librosa.power_to_db`
4. Normalize the spectrogram values to the range `[0, 1]`
5. Expand the single-channel spectrogram into a 3-channel RGB-like array
6. Resize the image to **150 × 150**
7. Convert the final array to `float32`

### Why this representation works

Mel-spectrograms preserve the structure of energy across time and frequency while remaining compact enough for image-based deep learning. The normalization step makes the data numerically stable, and the 3-channel representation allows the same preprocessing output to be used consistently by both the notebook pipeline and the FastAPI backend.

---

## 4. Model Architecture

### Siamese network design

The core model is a Siamese network with a shared convolutional encoder. Each of the two inputs is passed through the same encoder, and the resulting embeddings are compared with an L1 distance layer before the final similarity prediction.

The model path used in the repository is `embdmodel.keras`, which is loaded by the backend with `safe_mode=False`.

### Encoder structure

The encoder produces a **64-dimensional embedding vector**. Based on the notebook and documentation, the encoder includes:

- four convolutional blocks
- ReLU activations
- dropout regularization with a rate of 0.5
- pooling layers for spatial downsampling
- global max pooling at the end

This design compresses the spectrogram into a small dense vector that retains the most discriminative audio features.

### Similarity head

The Siamese comparator uses:

- shared encoder weights
- element-wise L1 distance between embeddings
- a final dense sigmoid classifier

The output is a similarity score in the range `[0, 1]`, where higher values indicate that the two segments are likely from the same song.

---

## 5. Training Implementation

The training workflow lives in [training.ipynb](training.ipynb). The notebook builds the encoder, wraps it in a Siamese architecture, and trains the network on generated pairs of spectrogram images.

### Pair generation

Training uses a batch generator that creates two types of examples:

- **positive pairs**: same-song pairs
- **negative pairs**: different-song pairs

The repository documentation indicates that batches are balanced, with approximately 50% positive and 50% negative pairs.

### Optimization setup

The notebook uses the following training configuration:

- optimizer: Adam
- loss: binary cross-entropy
- batch size: 10
- epochs: 50
- early stopping on validation loss
- model checkpointing to `embdmodel.keras`

The notebook also uses early stopping with patience and a minimum delta, which helps prevent overfitting and keeps the best-performing checkpoint.

### Training characteristics

The generator-based design is important because it avoids loading the entire training set into memory. Instead, spectrograms are created and consumed in batches, which is more memory efficient for audio workloads.

---

## 6. Inference and Reference Matching

The inference workflow is implemented in [test.ipynb](test.ipynb), [build_reference_index.py](build_reference_index.py), and [backend/app.py](backend/app.py).

### Embedding extraction

The backend loads `embdmodel.keras`, then extracts the encoder from `model.layers[2]`. That encoder is used to turn a query spectrogram into a **64-dimensional embedding**.

### Reference index format

The reference index is a pickle file containing the following structure:

```python
{song_name: [embedding_1, embedding_2, ...]}
```

The `build_reference_index.py` script creates this file from Jamendo MP3 tracks and metadata. For each audio file it:

- loads the full track
- splits it into 10-second segments
- generates a spectrogram input for each segment
- predicts the embedding for each segment
- stores all embeddings under the song name

### Matching strategy

At query time, the backend:

1. converts the uploaded file into a spectrogram input
2. predicts a query embedding
3. compares that embedding with all stored reference embeddings using L1 distance
4. selects the smallest distance per song
5. returns the top-k candidates and the best match

This makes recognition fast because the expensive embedding generation for the reference songs happens once during index building, not on every request.

---

## 7. FastAPI Backend

The backend service is implemented in [backend/app.py](backend/app.py).

### Main responsibilities

- load the saved Keras model
- expose a health endpoint
- convert uploaded audio into model input
- return raw embeddings
- perform recognition against the reference index

### Endpoints

- `GET /health`
  - returns service status, loaded model path, embedding shape, and the number of reference songs
- `POST /embed`
  - accepts an audio file and returns the extracted embedding vector
- `POST /recognize`
  - accepts an audio file and returns the best match plus ranked candidates

### Runtime behavior

The backend searches for `embdmodel.keras` in these places:

1. `backend/embdmodel.keras`
2. the project root
3. a custom path from the `MODEL_PATH` environment variable

Similarly, it looks for the reference pickle index in:

1. `REFERENCE_INDEX_PATH`
2. `backend/dict.pickle`
3. `dict.pickle` at the project root

If no reference index is available, `/recognize` returns a service-unavailable error rather than pretending to match against an empty database.

### Audio handling in the API

Uploaded audio is written to a temporary file, loaded through `librosa`, optionally trimmed to the first 10 seconds, converted to a mel-spectrogram, normalized, resized to 150×150, and expanded to a batch tensor before embedding prediction.

---

## 8. Streamlit Frontend

The frontend is implemented in [frontend/streamlit_app.py](frontend/streamlit_app.py).

### UI behavior

The app provides:

- a file uploader for MP3, WAV, M4A, and OGG files
- a sidebar to configure the backend base URL
- a slider for the number of top matches to return
- a recognition button that calls the backend `/recognize` endpoint
- a result panel that shows the best match, distance, embedding shape, and a ranked candidate table

### Visual design

The Streamlit app uses a custom dark gradient layout with a hero banner and styled result cards, giving the interface a more polished presentation than the default Streamlit theme.

---

## 9. Complexity and Performance

### Training complexity

Training is heavier because spectrograms are generated repeatedly as batches are produced. The overall cost scales with the number of samples, the number of batches, and the number of epochs.

### Inference complexity

Inference is significantly cheaper because:

- the reference embeddings are precomputed once
- each query requires only one forward pass through the encoder
- matching is reduced to vector distance calculations

In practice, this makes recognition manageable even as the reference library grows.

### Key optimization

The most important optimization in the project is the separation of:

- **feature extraction**, done once and cached
- **query-time comparison**, done on embeddings only

That design avoids repeated spectrogram generation and repeated full-model inference for every database song during recognition.

---

## 10. Implementation Files

The main files in the repository are:

- [training.ipynb](training.ipynb) - Siamese model training and pair generation
- [test.ipynb](test.ipynb) - embedding extraction and song matching workflow
- [build_reference_index.py](build_reference_index.py) - creates the cached reference embedding dictionary
- [jamendo_pipeline.py](jamendo_pipeline.py) - downloads data and generates splits
- [backend/app.py](backend/app.py) - FastAPI inference service
- [frontend/streamlit_app.py](frontend/streamlit_app.py) - Streamlit user interface
- [embdmodel.keras](embdmodel.keras) - saved trained model
- [jamendo_dataset/metadata.csv](jamendo_dataset/metadata.csv) - dataset manifest

---

## 11. Known Constraints

The implementation is functional, but it still has a few practical constraints:

- the backend depends on `safe_mode=False` when loading the Keras model
- recognition quality depends on having a valid reference index built from the same preprocessing assumptions
- the current matching strategy uses L1 distance only, so there is room for threshold tuning and evaluation metrics
- the repository documents the workflow, but it does not yet include a formal benchmark table for accuracy, precision, or recall

---

## 12. Conclusion

This project delivers an end-to-end music recognition pipeline built on metric learning. The Siamese CNN learns audio embeddings from mel-spectrograms, the Jamendo utilities support reproducible data preparation, the reference index enables efficient matching, and the FastAPI plus Streamlit stack makes the model usable as a simple application.

Overall, the implementation is organized around a practical deployment pattern: train once, cache embeddings, and recognize new audio through lightweight similarity search.