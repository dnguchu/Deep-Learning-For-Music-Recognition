# Deep-Learning-For-Music-Recognition
This is my final year project in my undergraduate Data Science and Analytics course.

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


Major Time Savings in Test
Phase	Training	Test
Spectrogram generation	Per epoch × batch loads (expensive)	Once upfront
Image I/O	Repeated disk access per training iteration	None (embeddings cached)
Forward passes	Full model per batch during training	Only during initial embedding computation
Query	N/A	O(m) simple distance calculations on vectors
