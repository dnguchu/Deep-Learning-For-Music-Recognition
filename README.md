# Deep-Learning-For-Music-Recognition
This is my final year project in my undergraduate Data Science and Analytics course.

## Serving the model

The model can be served with a FastAPI backend and a Streamlit frontend using the structure below:

```text
project/
│
├── backend/
│   ├── app.py
│   ├── embdmodel.keras
│   └── requirements.txt
│
├── frontend/
│   ├── streamlit_app.py
│   └── requirements.txt
│
└── README.md
```

### What the backend exposes

The backend loads `embdmodel.keras`, extracts the encoder at `model.layers[2]`, converts uploaded audio into a mel-spectrogram, and returns either:

* a raw 64-dimensional embedding via `POST /embed`
* a ranked recognition result via `POST /recognize`

If you have a cached song embedding index from the notebook workflow, set `REFERENCE_INDEX_PATH` to the pickle file before starting the API.

For the Jamendo files in `jamendo_dataset/downloaded_tracks`, you can build that index with:

```bash
python build_reference_index.py --audio-dir jamendo_dataset/downloaded_tracks --metadata-csv jamendo_dataset/metadata.csv --output dict.pickle
```

After that, start the backend with `REFERENCE_INDEX_PATH=dict.pickle` or place the file at the project root so the API finds it automatically.

## Detailed setup instructions

### 1. Create and activate a virtual environment

From the project root, activate the environment you want to use or create a fresh one:

```bash
python -m venv venv
source venv/bin/activate
```

If you already have an environment, activate it before installing any packages.

### 2. Install backend dependencies

The backend needs FastAPI, Uvicorn, TensorFlow, librosa, OpenCV, NumPy, and the multipart package used for file uploads.

```bash
```

### 3. Install frontend dependencies

The frontend is a separate Streamlit app that calls the backend over HTTP.

```bash
pip install -r frontend/requirements.txt
```

### 4. Place the saved model in the expected location

The backend automatically looks for the model in these locations:

1. `backend/embdmodel.keras`
2. `./embdmodel.keras` at the project root
3. A custom path provided through `MODEL_PATH`

The recommended layout is to copy the model into the backend folder:

```bash
cp embdmodel.keras backend/embdmodel.keras
```

If you prefer not to move the file, keep the root-level model in place and the backend will still load it.

### 5. Optional: provide a reference embedding index

The recognition endpoint is most useful when the backend can compare the uploaded song against a cached embedding dictionary generated from your notebook workflow.

If you have a pickle file such as `dict.pickle`, point the backend to it before starting the server:

```bash
export REFERENCE_INDEX_PATH=/full/path/to/dict.pickle
```

If this variable is not set, the API will still run and return embeddings, but song matching results will be empty because there is no reference index to compare against.

### 6. Start the FastAPI backend

Run the API from the project root:

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

What this does:

* starts the API on `http://localhost:8000`
* enables auto-reload during development
* exposes the `/health`, `/embed`, and `/recognize` endpoints

You can verify the server is running by opening:

```text
http://localhost:8000/health
```

### 7. Start the Streamlit frontend

Open a second terminal, keep the same virtual environment active, and run:

```bash
streamlit run frontend/streamlit_app.py
```

The app will open in your browser and send uploaded audio files to the backend.

### 8. Use the frontend

1. Upload an MP3, WAV, M4A, or OGG file.
2. Confirm the API base URL in the sidebar.
3. Choose how many top matches you want returned.
4. Click **Recognize song**.

The frontend displays the best match and a ranked table of candidate songs returned by the backend.

## Environment variables

The serving stack supports a few optional environment variables:

* `MODEL_PATH` - absolute path to `embdmodel.keras` if it is not in the default locations
* `REFERENCE_INDEX_PATH` - path to the pickle file containing cached song embeddings
* `API_BASE_URL` - backend URL used by Streamlit, defaults to `http://localhost:8000`

Example:

```bash
export MODEL_PATH=/mnt/Data/Projects/Deep-Learning-For-Music-Recognition/backend/embdmodel.keras
export REFERENCE_INDEX_PATH=/mnt/Data/Projects/Deep-Learning-For-Music-Recognition/dict.pickle
export API_BASE_URL=http://localhost:8000
```

## Troubleshooting

If the backend fails to start, check these common issues:

* `embdmodel.keras` is missing or not in one of the supported locations
* TensorFlow or another dependency is not installed in the active environment
* The reference index path is wrong or points to a file that does not exist

If the frontend cannot reach the backend, confirm that:

* Uvicorn is still running on port `8000`
* `API_BASE_URL` points to the correct host and port
* the browser is not blocking local requests

## Notes

* The API looks for `embdmodel.keras` in `backend/` first, then in the project root.
* The frontend expects the API at `http://localhost:8000` unless `API_BASE_URL` is set.
* Recognition quality depends on having a reference embedding index available to the backend.

#Efficiency comparison
Time Complexity Comparison
Training.ipynb - Training Phase
Complexity: O(n × m × epochs)

Generates spectrograms on-the-fly during training via batch generator
Each epoch processes every training sample with image loading and preprocessing happening in the batch generator callback
Key inefficiency: Spectrograms are created from raw audio, saved to disk, then loaded repeatedly during training—disk I/O per epoch
Test.ipynb - Inference Phase
Complexity: O(n + m)

Pre-computed embeddings: Test songs are processed once upfront:
Split into 10s segments
Create spectrograms (once per segment)
Generate embeddings immediately (not image loading—directly from model.predict)
Store embeddings in dictionary
Query matching: O(m) single pass through pre-computed embeddings with vector distance calculation
Key optimization: Eliminates redundant spectrogram generation and disk I/O—embeddings cached in memory


Jamendo-backed data preparation
The new `jamendo_pipeline.py` utility adds a Jamendo API download path, metadata manifest generation, and train/validation/test split creation so the notebooks can be pointed at a reproducible dataset instead of a fixed local music folder.


Major Time Savings in Test
Phase	Training	Test
Spectrogram generation	Per epoch × batch loads (expensive)	Once upfront
Image I/O	Repeated disk access per training iteration	None (embeddings cached)
Forward passes	Full model per batch during training	Only during initial embedding computation
Query	N/A	O(m) simple distance calculations on vectors
