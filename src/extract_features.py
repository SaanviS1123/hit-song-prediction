import librosa
import numpy as np
from scipy.stats import skew, kurtosis


def extract_features(audio_file):

    audio, sr = librosa.load(audio_file, sr=None)

    features = {}

    features["chroma_stft"] = librosa.feature.chroma_stft(y=audio, sr=sr)
    features["chroma_cqt"] = librosa.feature.chroma_cqt(y=audio, sr=sr)
    features["chroma_cens"] = librosa.feature.chroma_cens(y=audio, sr=sr)
    features["mfcc"] = librosa.feature.mfcc(y=audio, sr=sr)
    features["rms"] = librosa.feature.rms(y=audio)
    features["spectral_centroid"] = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features["spectral_contrast"] = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features["spectral_rolloff"] = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features["tonnetz"] = librosa.feature.tonnetz(y=audio, sr=sr)
    features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(y=audio)

    feature_vector = []

    for name, feature in features.items():

        for row in feature:
            feature_vector.extend([
                np.min(row),
                np.mean(row),
                np.median(row),
                np.max(row),
                np.std(row),
                skew(row),
                kurtosis(row)
            ])

    return feature_vector


# features = extract_features("data/chorus/test_chorus.wav")

# print("Total features:", len(features))