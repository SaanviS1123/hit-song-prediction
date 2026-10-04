import librosa

audio, sr = librosa.load("data/raw/test.wav", sr=None)

print("Audio loaded successfully!")
print("Sample rate:", sr)
print("Number of samples:", len(audio))
print("Duration:", len(audio) / sr, "seconds")