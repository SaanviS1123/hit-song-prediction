import librosa

audio, sr = librosa.load("data/chorus/test_chorus.wav", sr=None)

mfcc = librosa.feature.mfcc(y=audio, sr=sr)

print("MFCC extracted successfully!")
print("Shape:", mfcc.shape)