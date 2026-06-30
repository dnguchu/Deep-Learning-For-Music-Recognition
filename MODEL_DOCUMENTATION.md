# Deep Learning for Music Recognition - Model Documentation

## Project Overview
This project implements a **Siamese Neural Network** for music recognition that identifies songs by comparing audio fingerprints. The system learns to recognize songs by analyzing 10-second audio segments converted to mel-spectrogram images.

---

## Model Architecture

### Core Components

#### 1. **Encoder Network**
- **Purpose**: Extracts meaningful features from spectrogram images
- **Architecture**:
  - Input: 150×150×3 RGB spectrogram image
  - Layer 1: 32 filters, 3×3 kernel, ReLU activation + Dropout(0.5)
  - Layer 2: 64 filters, 3×3 kernel, ReLU activation + MaxPool(2×2) + Dropout(0.5)
  - Layer 3: 64 filters, 3×3 kernel, ReLU activation + Dropout(0.5)
  - Layer 4: 64 filters, 3×3 kernel, ReLU activation + MaxPool(2×2) + Dropout(0.5)
  - Output Layer: Global Max Pooling → 64-dimensional embedding vector
- **Output**: 64-dimensional feature representation of audio

#### 2. **Siamese Network**
- **Purpose**: Compares two spectrogram embeddings to determine if they're from the same song
- **Architecture**:
  - Input: Two 150×150×3 spectrogram images (input1, input2)
  - Both inputs pass through the **same encoder** network (weight-shared)
  - L1 Distance Layer: Calculates element-wise absolute difference between embeddings
    - Output shape: (None, 64)
  - Classification Layer: Dense(1, sigmoid) → Binary output (0-1)
- **Training**: Binary cross-entropy loss to learn similarity

---

## How It Works

### Training Phase
```
Audio File → Split into 10-second segments → Generate mel-spectrograms → 
Siamese pairs (same/different labels) → Train with batch generator
```

**Batch Generation Strategy**:
- **Positive Pairs** (50% of batch): Same spectrogram paired with itself (label=1)
- **Negative Pairs** (50% of batch): Two different spectrograms (label=0)
- Batch Size: 10
- Epochs: 50 (with early stopping on validation loss, patience=10)

**Why Siamese?**
- Learns a metric space where similar songs have small embedding distances
- Efficient comparison: Only requires comparing two embeddings, not classifying each individually

### Testing/Inference Phase
```
Database Songs → Split into segments → Generate spectrograms → 
Create embeddings (once, cached) → Store in dictionary

Query Song → Generate spectrogram → Create embedding → 
Compare L1 distance to all database embeddings → Return best match
```

**Key Optimization**:
- Embeddings computed once and cached in memory (not repeated)
- Query requires only a single forward pass + vector distance calculations

---

## Audio Processing Pipeline

### 1. **Spectrogram Generation**
- **Input**: Raw audio file (MP3/WAV)
- **Process**:
  - Load using librosa with original sample rate
  - Extract mel-spectrogram (frequency-time representation of audio)
  - Convert to dB scale: `librosa.power_to_db(S, ref=np.max)`
  - Save as PNG image (400 DPI, 72×72 pixels)
- **Output**: Visual representation of audio frequency content
- **Why Mel-Spectrograms?**: Compressed frequency scale mimics human hearing perception

### 2. **Image Preprocessing**
- Load spectrogram PNG
- Convert BGR→RGB color space
- Resize to 150×150 pixels (model input requirement)
- Normalize: Divide by 255.0 (pixel values 0-1)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 10 |
| Epochs | 50 |
| Early Stopping | Patience=10, min_delta=0.0001 |
| Train/Test Split | 75% / 25% |
| Model Checkpoint | Saves best model (lowest val_loss) as `embdmodel.keras` |

**Dropout Strategy**: 0.5 dropout after each conv layer to prevent overfitting

---

## Testing & Song Recognition

### Database Creation (test.ipynb - Cell 4)
1. Process all songs in music library
2. For each song:
   - Split into 10-second segments
   - Generate spectrogram for each segment
   - Create embedding using trained encoder
   - Store all embeddings in `songspecdict` dictionary
3. Save embeddings to `dict.pickle` for reuse

### Song Matching (test.ipynb - Cell 6)
1. Load query audio (10-second segment)
2. Generate spectrogram and embedding
3. Calculate L1 distance to all cached embeddings:
   - `distance = ||query_embedding - database_embedding||₁`
4. Find minimum distance
5. Return corresponding song title

**Time Complexity**:
- Query: O(m) where m = total segments across all songs
- No spectrogram regeneration or disk I/O needed

---

## Model Inputs & Outputs

### Training Input
- **Two spectrogram images** (150×150×3 each)
- **Label**: 1 (same song) or 0 (different songs)

### Training Output
- **Similarity score**: 0-1 (probability of being from same song)

### Inference Input
- **Spectrogram image** (150×150×3)

### Inference Output
- **64-dimensional embedding vector** (feature representation)
- Use for distance comparison with database

---

## Current Model Status

✅ **Completed**:
- Encoder architecture implemented and trained
- Siamese network structure functional
- Audio preprocessing pipeline working
- Training with batch generation (avoids RAM overload)
- Embedding-based inference system operational
- Song recognition pipeline functional

📊 **Model Output**: 
- Saves as `embdmodel.keras` (Keras 3 format)
- Contains full siamese network (encoder + comparison layers)
- Ready for inference and music recognition tasks

---

## Key Features

1. **Metric Learning**: Learns similarity in embedding space rather than explicit classification
2. **Efficient Inference**: Pre-computed embeddings eliminate redundant computation
3. **Scalable**: New songs added by computing embeddings once and appending to dictionary
4. **Robust to Variations**: Learns from paired comparisons, not absolute features
5. **Memory Efficient**: Batch generation avoids loading entire dataset into RAM

---

## Data Flow Diagram

```
TRAINING PHASE:
Audio Files
    ↓
Segment (10s)
    ↓
Spectrogram (PNG)
    ↓
Preprocess (150×150)
    ↓
Batch Generator (Pairs)
    ↓
Siamese Network
    ↓
Binary Cross-Entropy Loss
    ↓
✓ Model Saved (embdmodel.keras)

TESTING PHASE:
Database Songs → Embeddings → dict.pickle

Query Audio
    ↓
Spectrogram (PNG)
    ↓
Preprocess (150×150)
    ↓
Encoder Forward Pass → Embedding
    ↓
L1 Distance to all cached embeddings
    ↓
Min Distance → Recognized Song
```

---

## Performance Considerations

| Aspect | Optimization |
|--------|--------------|
| Training Memory | Batch generator processes data in chunks |
| Inference Speed | Pre-computed embeddings eliminate redundant forward passes |
| Storage | Dictionary pickle (~MB) vs. full images (GB) |
| Scalability | Linear with number of songs; constant per-query time |
| Accuracy | Learning from pairs improves generalization |

---

## Dependencies

- **TensorFlow/Keras**: Deep learning framework
- **librosa**: Audio processing and spectrogram generation
- **OpenCV (cv2)**: Image preprocessing
- **NumPy**: Numerical operations
- **Matplotlib**: Visualization and spectrogram rendering
- **scikit-learn**: Train-test splitting and utilities

---

## Files

- `training.ipynb`: Model training pipeline (encoder + siamese network)
- `test.ipynb`: Inference, embedding generation, and song recognition
- `embdmodel.keras`: Trained model checkpoint
- `dict.pickle`: Cached embeddings for database songs
