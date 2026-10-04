# Predicting Hit Songs Using Audio Features

## Project Overview

This mini-project focuses on predicting whether a song is a hit or a non-hit using machine learning and audio features.

The project is based on the approach presented in the reference paper **"Predicting Hit Songs Using Repeated Chorus"**. The pipeline extracts the repeated chorus of a song, computes audio features from the chorus, and uses these features to train classification models.

The complete pipeline consists of:

1. Song dataset collection
2. Hit/non-hit labeling
3. Chorus extraction
4. Audio feature extraction
5. Feature preprocessing
6. Feature selection and PCA
7. Machine learning model training
8. 5-fold cross-validation
9. Test-set evaluation
10. ROC-AUC analysis
11. Feature-importance analysis

---

## Dataset

The current dataset contains **50 songs**:

- **26 hit songs**
- **24 non-hit songs**

Each song is represented using audio features extracted from its chorus.

The final feature dataset is stored in:

`data/features/features.csv`

The dataset contains **518 audio features per song**, along with the artist, song name, and hit/non-hit label.

### Labels

- `1` → Hit song
- `0` → Non-hit song

---

## Audio Feature Extraction

The repeated chorus of each song is extracted before feature calculation.

The following 11 types of audio features are extracted using **Librosa**:

- Chroma STFT
- Chroma CQT
- Chroma CENS
- MFCC
- RMS Energy
- Spectral Centroid
- Spectral Bandwidth
- Spectral Contrast
- Spectral Rolloff
- Tonnetz
- Zero Crossing Rate

For each feature dimension, the following seven statistical measures are calculated:

- Minimum
- Mean
- Median
- Maximum
- Standard Deviation
- Skewness
- Kurtosis

This produces **518 numerical audio features per song**.

---

## Machine Learning

Four classification models are implemented:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)

The machine learning pipeline includes:

- Data loading and validation
- Feature preprocessing
- Feature selection
- Standardization
- Principal Component Analysis (PCA)
- Stratified 5-fold cross-validation
- Model training
- Test-set prediction

The trained models are saved in the `models/` directory.

---

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The project also generates:

- Confusion matrices
- ROC curves
- Model comparison plots
- Feature-importance analysis

The generated results are stored in the `results/` directory.

---

## Project Structure

```text
hit-song-prediction/
│
├── data/
│   ├── songs.csv
│   ├── non_hit_candidates.csv
│   ├── final_songs.csv
│   └── features/
│       └── features.csv
│
├── models/
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
│
├── results/
│   ├── confusion_matrices/
│   ├── plots/
│   ├── evaluation_results.csv
│   ├── feature_importance.csv
│   └── model_comparison.csv
│
├── src/
│   ├── extract_features.py
│   ├── process_dataset.py
│   ├── process_song.py
│   ├── get_billboard_songs.py
│   ├── get_non_hit_songs.py
│   ├── create_final_dataset.py
│   └── ml/
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── train_models.py
│       ├── evaluate_models.py
│       └── feature_importance.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the Repository

git clone https://github.com/SaanviS1123/hit-song-prediction.git
cd hit-song-prediction

### 2. Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate

### 3. Install Dependencies

pip install -r requirements.txt

## Running the Project

### Audio Feature Extraction

To process the audio dataset and generate the feature dataset:

```bash
python src/process_dataset.py
```

The generated feature dataset will be saved to:

data/features/features.csv
Machine Learning

The machine learning scripts are located in:
src/ml/

The trained models are saved in:
models/

Evaluation results and visualizations are saved in:
results/

Results
The project evaluates the trained models using accuracy, precision, recall, F1-score, and ROC-AUC.

The evaluation results are available in:
results/evaluation_results.csv

Model comparison results are available in:
results/model_comparison.csv

Limitations
The current dataset contains 50 songs, which is smaller than the dataset used in the reference paper. Therefore, the results should be considered a small-scale implementation of the proposed approach.

Conclusion
This project demonstrates an end-to-end machine learning pipeline for predicting hit songs using audio characteristics extracted from repeated choruses. It combines audio processing, statistical feature engineering, dimensionality reduction, and machine learning classification.

Reference
"Predicting Hit Songs Using Repeated Chorus" — CS229 Milestone Report
