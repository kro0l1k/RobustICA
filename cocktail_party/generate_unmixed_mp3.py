import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

# Load the numpy array from the file
file_name = "S_hat_rica.npy"
data = np.load(file_name)

# Validate that the array has 3 columns
if data.shape[1] != 3:
    raise ValueError("The numpy array must have exactly 3 columns.")

# Sampling rate for the audio (44.1 kHz is standard for CD-quality audio)
sample_rate = 2 * 44100 # NOTE: add the 2 multiplier for music. dont do it for poertry.

# Duration of each sample point in seconds (inverse of the sampling rate)
duration = 1 / sample_rate

# Generate audio files for each column
for i in range(3):
    # Normalize the data to be within the range of -1.0 to 1.0
    signal = data[:, i] / np.max(np.abs(data[:, i]))

    # Convert the normalized signal into PCM (Pulse Code Modulation) format
    samples = (signal * 32767).astype(np.int16)

    # Create a raw audio segment
    audio_segment = AudioSegment(
        samples.tobytes(), 
        frame_rate=sample_rate, 
        sample_width=2,  # 2 bytes per sample (16-bit PCM)
        channels=1       # Mono audio
    )

    # Export the audio to an MP3 file
    output_file = f"music{i+1}rica.mp3"
    audio_segment.export(output_file, format="mp3")
    print(f"Generated {output_file}")