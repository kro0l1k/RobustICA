import numpy as np
from pydub import AudioSegment
import os
print(os.path.exists("gray.mp3"))
print(os.path.exists("goethe.mp3"))
print(os.path.exists("kordian.mp3"))

# Load the MP3 files
audio_gray = AudioSegment.from_mp3("gray.mp3")
audio_goethe = AudioSegment.from_mp3("goethe.mp3")
audio_kordian = AudioSegment.from_mp3("kordian.mp3")

# Convert each to a NumPy array of samples
data_gray = np.array(audio_gray.get_array_of_samples(), dtype=np.float32)
data_goethe = np.array(audio_goethe.get_array_of_samples(), dtype=np.float32)
data_kordian = np.array(audio_kordian.get_array_of_samples(), dtype=np.float32)

# Ensure they are all the same length
# If not, you might need to truncate or pad them.
min_length = min(len(data_gray), len(data_goethe), len(data_kordian))
data_gray = data_gray[:min_length]
data_goethe = data_goethe[:min_length]
data_kordian = data_kordian[:min_length]

# Stack into a single S matrix of shape (N, 3)
S = np.column_stack((data_gray, data_goethe, data_kordian))

# Define the matrix A
A = np.array([[1,  2, 0],
              [0, 1,  2],
              [0, 0, 1]])

# Compute X = S * A^T
X = S @ A.T  # Matrix multiplication

X = X.astype(np.float32)  # Ensure the data type is float32 for audio processing
X = X[:5_000_000, :]  # Use only the first 5,000,000 samples for consistency
# Save the result as a numpy file
np.save("observation_cocktail_party.npy", X)

# print out the sample rate and the number of samples
print(audio_gray.frame_rate)
print(len(data_gray))
print(audio_goethe.frame_rate)
print(len(data_goethe))
print(audio_kordian.frame_rate)
print(len(data_kordian))
print(X.shape)
