# Python source code exported from cancer-project-ann-62f017.ipynb
# Original Kaggle notebook code preserved in cell order.

# %% [Cell 0]
#import
import os
import shutil
import pprint
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# %% [Cell 1]
df_meta = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/meta.csv')
df_meta.head()

# %% [Cell 2]
# load dicom info file
df_dicom = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv')
df_dicom.head()

# %% [Cell 3]
# check image types in dataset
df_dicom.SeriesDescription.unique()

# %% [Cell 4]
# check image path in dataset
# cropped images
cropped_images = df_dicom[df_dicom.SeriesDescription=='cropped images'].image_path
#cropped_images.head(5)

# %% [Cell 5]
#full mammogram images
full_mammo = df_dicom[df_dicom.SeriesDescription=='full mammogram images'].image_path
#full_mammo.head(5)

# %% [Cell 6]
# ROI images
roi_img = df_dicom[df_dicom.SeriesDescription=='ROI mask images'].image_path
#roi_img.head(5)

# %% [Cell 7]
print("Example full mammogram paths:")
print(full_mammo.head(5).tolist())

print("\nExample cropped image paths:")
print(cropped_images.head(5).tolist())

print("\nExample ROI image paths:")
print(roi_img.head(5).tolist())

# %% [Cell 8]
import pandas as pd

# Load DICOM metadata
df_dicom = pd.read_csv(
    '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv'
)

# Create image-path lists
cropped_images = df_dicom[
    df_dicom['SeriesDescription'] == 'cropped images'
]['image_path']

full_mammo = df_dicom[
    df_dicom['SeriesDescription'] == 'full mammogram images'
]['image_path']

roi_img = df_dicom[
    df_dicom['SeriesDescription'] == 'ROI mask images'
]['image_path']

print("DICOM rows:", len(df_dicom))
print("Full mammogram paths:", len(full_mammo))
print("Cropped image paths:", len(cropped_images))
print("ROI image paths:", len(roi_img))

# %% [Cell 9]
print("Example full mammogram paths:")
print(full_mammo.head(5).tolist())

print("\nExample cropped image paths:")
print(cropped_images.head(5).tolist())

print("\nExample ROI image paths:")
print(roi_img.head(5).tolist())

# %% [Cell 10]
from pathlib import Path

DATASET_ROOT = Path(
    "/kaggle/input/cbis-ddsm-breast-cancer-image-dataset"
)

def make_image_path(relative_path):
    if pd.isna(relative_path):
        return None
    
    relative_path = str(relative_path)
    
    relative_path = relative_path.replace(
        "CBIS-DDSM/jpeg/",
        ""
    )
    
    return DATASET_ROOT / "jpeg" / relative_path

test_path = make_image_path(full_mammo.iloc[0])

print("Original path:")
print(full_mammo.iloc[0])

print("\nConverted Kaggle path:")
print(test_path)

print("\nDoes the image exist?")
print(test_path.exists())

# %% [Cell 11]
# Create actual image paths for the mass datasets

def convert_dataset_image_path(path):
    if pd.isna(path):
        return None

    path = str(path)

    # The CSV paths begin with CBIS-DDSM/jpeg/
    path = path.replace("CBIS-DDSM/jpeg/", "")

    return str(DATASET_ROOT / "jpeg" / path)


mass_train["full_image_path"] = mass_train["image file path"].apply(
    convert_dataset_image_path
)

mass_test["full_image_path"] = mass_test["image file path"].apply(
    convert_dataset_image_path
)

print("Training image paths created:", mass_train["full_image_path"].notna().sum())
print("Testing image paths created:", mass_test["full_image_path"].notna().sum())

print("\nExample training image path:")
print(mass_train["full_image_path"].iloc[0])

# %% [Cell 12]
# Reload the cancer diagnosis datasets

mass_train = pd.read_csv(
    '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv'
)

mass_test = pd.read_csv(
    '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv'
)

# Recreate the dataset root
DATASET_ROOT = Path(
    "/kaggle/input/cbis-ddsm-breast-cancer-image-dataset"
)

# Convert CBIS-DDSM paths to actual Kaggle paths
def convert_dataset_image_path(path):
    if pd.isna(path):
        return None

    path = str(path)
    path = path.replace("CBIS-DDSM/jpeg/", "")

    return str(DATASET_ROOT / "jpeg" / path)

# Create full mammogram paths
mass_train["full_image_path"] = mass_train["image file path"].apply(
    convert_dataset_image_path
)

mass_test["full_image_path"] = mass_test["image file path"].apply(
    convert_dataset_image_path
)

print("Training records:", len(mass_train))
print("Testing records:", len(mass_test))

print("\nTraining image paths created:",
      mass_train["full_image_path"].notna().sum())

print("Testing image paths created:",
      mass_test["full_image_path"].notna().sum())

print("\nExample training image path:")
print(mass_train["full_image_path"].iloc[0])

# %% [Cell 13]
from pathlib import Path

# Check whether the first training image exists
first_training_path = Path(mass_train["full_image_path"].iloc[0])

print("Training image path:")
print(first_training_path)

print("\nDoes this file exist?")
print(first_training_path.exists())

print("\nFile extension:")
print(first_training_path.suffix)

# %% [Cell 14]
# Match the DICOM paths from the cancer dataset
# to the corresponding JPEG paths in dicom_info.csv

dicom_lookup = df_dicom.copy()

# Remove the common prefix so both datasets use the same path format
dicom_lookup["dicom_key"] = dicom_lookup["file_path"].astype(str).str.replace(
    "CBIS-DDSM/dicom/",
    "",
    regex=False
)

# Create a lookup: DICOM path -> JPEG path
dicom_to_jpeg = dict(
    zip(
        dicom_lookup["dicom_key"],
        dicom_lookup["image_path"]
    )
)

# Find the corresponding JPEG path for each cancer record
mass_train["jpeg_image_path"] = mass_train["image file path"].astype(str).map(
    dicom_to_jpeg
)

mass_test["jpeg_image_path"] = mass_test["image file path"].astype(str).map(
    dicom_to_jpeg
)

print("Training records with JPEG paths:",
      mass_train["jpeg_image_path"].notna().sum())

print("Testing records with JPEG paths:",
      mass_test["jpeg_image_path"].notna().sum())

print("\nExample matched JPEG path:")
print(mass_train["jpeg_image_path"].iloc[0])

# %% [Cell 15]
print("FIRST MASS TRAIN DICOM PATH:")
print(mass_train["image file path"].iloc[0])

print("\nFIRST DICOM_INFO FILE PATH:")
print(df_dicom["file_path"].iloc[0])

print("\nFIRST DICOM_INFO JPEG PATH:")
print(df_dicom["image_path"].iloc[0])

print("\n--- LAST 2 PARTS OF EACH PATH ---")

print("\nMass train:")
print("/".join(str(mass_train["image file path"].iloc[0]).split("/")[-2:]))

print("\nDICOM info:")
print("/".join(str(df_dicom["file_path"].iloc[0]).split("/")[-2:]))

# %% [Cell 16]
# Extract the Series UID from each dataset

mass_train["series_uid"] = (
    mass_train["image file path"]
    .astype(str)
    .str.rstrip("/")
    .str.split("/")
    .str[-2]
)

mass_test["series_uid"] = (
    mass_test["image file path"]
    .astype(str)
    .str.rstrip("/")
    .str.split("/")
    .str[-2]
)

df_dicom["series_uid"] = (
    df_dicom["file_path"]
    .astype(str)
    .str.rstrip("/")
    .str.split("/")
    .str[-2]
)

# Look only at full mammogram images
full_mammo_info = df_dicom[
    df_dicom["SeriesDescription"] == "full mammogram images"
].copy()

print("First training Series UID:")
print(mass_train["series_uid"].iloc[0])

print("\nNumber of full mammogram metadata records:")
print(len(full_mammo_info))

print("\nTraining Series UIDs matched:")
print(
    mass_train["series_uid"]
    .isin(set(full_mammo_info["series_uid"]))
    .sum()
)

print("\nTesting Series UIDs matched:")
print(
    mass_test["series_uid"]
    .isin(set(full_mammo_info["series_uid"]))
    .sum()
)

print("\nMatching metadata record for the first training case:")
print(
    full_mammo_info[
        full_mammo_info["series_uid"] == mass_train["series_uid"].iloc[0]
    ][["file_path", "image_path", "SeriesDescription"]]
)

# %% [Cell 17]
# Check whether each Series UID appears only once
print(
    "Duplicate full-mammogram Series UIDs:",
    full_mammo_info["series_uid"].duplicated().sum()
)

# Create a lookup from Series UID to the JPEG image path
series_to_jpeg = dict(
    zip(
        full_mammo_info["series_uid"],
        full_mammo_info["image_path"]
    )
)

# Attach the JPEG path to the training and testing datasets
mass_train["jpeg_image_path"] = mass_train["series_uid"].map(series_to_jpeg)
mass_test["jpeg_image_path"] = mass_test["series_uid"].map(series_to_jpeg)

print("Training JPEG paths found:", mass_train["jpeg_image_path"].notna().sum())
print("Testing JPEG paths found:", mass_test["jpeg_image_path"].notna().sum())

print("\nExample JPEG path:")
print(mass_train["jpeg_image_path"].iloc[0])

# %% [Cell 18]
# Convert the dataset JPEG paths into actual Kaggle file paths

DATASET_ROOT = Path("/kaggle/input/cbis-ddsm-breast-cancer-image-dataset")

def jpeg_to_local_path(path):
    if pd.isna(path):
        return None
    
    path = str(path).replace("CBIS-DDSM/jpeg/", "")
    return str(DATASET_ROOT / "jpeg" / path)

mass_train["local_image_path"] = mass_train["jpeg_image_path"].apply(
    jpeg_to_local_path
)

mass_test["local_image_path"] = mass_test["jpeg_image_path"].apply(
    jpeg_to_local_path
)

# Check whether the actual image files exist
mass_train["image_exists"] = mass_train["local_image_path"].apply(
    lambda x: Path(x).exists() if x else False
)

mass_test["image_exists"] = mass_test["local_image_path"].apply(
    lambda x: Path(x).exists() if x else False
)

print("Training images that exist:",
      mass_train["image_exists"].sum(), "/", len(mass_train))

print("Testing images that exist:",
      mass_test["image_exists"].sum(), "/", len(mass_test))

print("\nExample local image path:")
print(mass_train["local_image_path"].iloc[0])

# %% [Cell 19]
# Load and display one mammogram image

from PIL import Image

sample_path = mass_train["local_image_path"].iloc[0]

sample_image = Image.open(sample_path)

print("Image size:", sample_image.size)
print("Image mode:", sample_image.mode)

plt.figure(figsize=(8, 8))
plt.imshow(sample_image, cmap="gray")
plt.axis("off")
plt.title("Sample CBIS-DDSM Mammogram")
plt.show()

# %% [Cell 20]
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.imshow(sample_image, cmap="gray")
plt.axis("off")
plt.title("Sample CBIS-DDSM Mammogram")
plt.show()

# %% [Cell 21]
# Check the diagnosis labels

print("Training pathology labels:")
print(mass_train["pathology"].value_counts())

print("\nTesting pathology labels:")
print(mass_test["pathology"].value_counts())

# %% [Cell 22]
# Create binary diagnosis labels
# 1 = Malignant
# 0 = Benign or Benign Without Callback

label_map = {
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0
}

mass_train["target"] = mass_train["pathology"].map(label_map)
mass_test["target"] = mass_test["pathology"].map(label_map)

print("Training target distribution:")
print(mass_train["target"].value_counts().sort_index())

print("\nTesting target distribution:")
print(mass_test["target"].value_counts().sort_index())

print("\nMissing training targets:", mass_train["target"].isna().sum())
print("Missing testing targets:", mass_test["target"].isna().sum())

# %% [Cell 23]
# Image preprocessing settings

IMG_SIZE = (96, 96)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize(IMG_SIZE)
    
    # Convert image to NumPy array and normalize pixels to 0–1
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    return image_array


# Test preprocessing on one image
sample_processed = preprocess_image(mass_train["local_image_path"].iloc[0])

print("Processed image shape:", sample_processed.shape)
print("Minimum pixel value:", sample_processed.min())
print("Maximum pixel value:", sample_processed.max())

# %% [Cell 24]
import numpy as np

# Run the preprocessing test again
sample_processed = preprocess_image(mass_train["local_image_path"].iloc[0])

print("Processed image shape:", sample_processed.shape)
print("Minimum pixel value:", sample_processed.min())
print("Maximum pixel value:", sample_processed.max())

# %% [Cell 25]
# Process all training and testing mammogram images

X_train = np.stack(
    mass_train["local_image_path"].apply(preprocess_image).values
)

X_test = np.stack(
    mass_test["local_image_path"].apply(preprocess_image).values
)

y_train = mass_train["target"].to_numpy()
y_test = mass_test["target"].to_numpy()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# %% [Cell 26]
# Create a patient-level training/validation split

from sklearn.model_selection import GroupShuffleSplit

groups = mass_train["patient_id"].to_numpy()

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X_train, y_train, groups=groups)
)

X_train_split = X_train[train_idx]
X_val = X_train[val_idx]

y_train_split = y_train[train_idx]
y_val = y_train[val_idx]

print("Training images:", X_train_split.shape)
print("Validation images:", X_val.shape)

print("\nTraining labels:")
print(np.bincount(y_train_split))

print("\nValidation labels:")
print(np.bincount(y_val))

print(
    "\nPatients in training:",
    mass_train.iloc[train_idx]["patient_id"].nunique()
)

print(
    "Patients in validation:",
    mass_train.iloc[val_idx]["patient_id"].nunique()
)

print(
    "Patients appearing in both:",
    len(
        set(mass_train.iloc[train_idx]["patient_id"])
        &
        set(mass_train.iloc[val_idx]["patient_id"])
    )
)

# %% [Cell 27]
# Add the grayscale channel dimension for TensorFlow

X_train_split = X_train_split[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("Training input shape:", X_train_split.shape)
print("Validation input shape:", X_val.shape)
print("Testing input shape:", X_test.shape)

# %% [Cell 28]
# Build the Artificial Neural Network (ANN)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

ann_model = Sequential([
    Input(shape=(96, 96, 1)),
    
    Flatten(),
    
    Dense(256, activation="relu"),
    Dropout(0.40),
    
    Dense(128, activation="relu"),
    Dropout(0.30),
    
    Dense(64, activation="relu"),
    Dropout(0.20),
    
    Dense(1, activation="sigmoid")
])

ann_model.summary()

# %% [Cell 29]
# Compile the ANN

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC

ann_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        BinaryAccuracy(name="accuracy"),
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ]
)

print("ANN compiled successfully.")

# %% [Cell 30]
# Set up training callbacks

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    mode="min",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    mode="min",
    min_delta=0.001,
    min_lr=0.000001,
    verbose=1
)

print("Training callbacks are ready.")

# %% [Cell 31]
# Train the ANN

history = ann_model.fit(
    X_train_split,
    y_train_split,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\nTraining completed.")
print("Epochs actually completed:", len(history.history["loss"]))

# %% [Cell 32]
# Evaluate the trained ANN on the untouched test set

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Generate predicted probabilities
y_test_prob = ann_model.predict(X_test, verbose=0).ravel()

# Convert probabilities to binary predictions
y_test_pred = (y_test_prob >= 0.5).astype(int)

# Calculate evaluation metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_test_prob)

print("TEST SET RESULTS")
print("=" * 40)
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-score : {test_f1:.4f}")
print(f"ROC-AUC  : {test_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["Non-Malignant", "Malignant"],
        zero_division=0
    )
)

# %% [Cell 33]
# Visualize the confusion matrix

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Malignant", "Malignant"],
    yticklabels=["Non-Malignant", "Malignant"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("ANN Confusion Matrix - CBIS-DDSM Test Set")
plt.show()

# %% [Cell 34]
# Plot training and validation loss

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


# Plot training and validation accuracy

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# %% [Cell 35]
# Build an improved, regularized ANN

from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

improved_model = Sequential([
    Input(shape=(96, 96, 1)),
    
    Flatten(),
    
    Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.0001)
    ),
    BatchNormalization(),
    Dropout(0.40),
    
    Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.0001)
    ),
    BatchNormalization(),
    Dropout(0.30),
    
    Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.0001)
    ),
    Dropout(0.20),
    
    Dense(1, activation="sigmoid")
])

improved_model.summary()

# %% [Cell 36]
# Compile the improved ANN

improved_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        BinaryAccuracy(name="accuracy"),
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ]
)

print("Improved ANN compiled successfully.")

# %% [Cell 37]
# Train the improved ANN

improved_history = improved_model.fit(
    X_train_split,
    y_train_split,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\nImproved ANN training completed.")
print(
    "Epochs actually completed:",
    len(improved_history.history["loss"])
)

# %% [Cell 38]
# Evaluate the improved ANN on the same untouched test set

improved_test_prob = improved_model.predict(
    X_test,
    verbose=0
).ravel()

improved_test_pred = (
    improved_test_prob >= 0.5
).astype(int)

improved_accuracy = accuracy_score(
    y_test,
    improved_test_pred
)

improved_precision = precision_score(
    y_test,
    improved_test_pred,
    zero_division=0
)

improved_recall = recall_score(
    y_test,
    improved_test_pred,
    zero_division=0
)

improved_f1 = f1_score(
    y_test,
    improved_test_pred,
    zero_division=0
)

improved_auc = roc_auc_score(
    y_test,
    improved_test_prob
)

print("IMPROVED ANN — TEST SET RESULTS")
print("=" * 45)
print(f"Accuracy : {improved_accuracy:.4f}")
print(f"Precision: {improved_precision:.4f}")
print(f"Recall   : {improved_recall:.4f}")
print(f"F1-score : {improved_f1:.4f}")
print(f"ROC-AUC  : {improved_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, improved_test_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        improved_test_pred,
        target_names=["Non-Malignant", "Malignant"],
        zero_division=0
    )
)

# %% [Cell 39]
# Compare baseline and improved ANN performance

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Baseline ANN": [
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        test_auc
    ],
    "Improved ANN": [
        improved_accuracy,
        improved_precision,
        improved_recall,
        improved_f1,
        improved_auc
    ]
})

comparison["Change"] = (
    comparison["Improved ANN"] -
    comparison["Baseline ANN"]
)

print(comparison.round(4))

# %% [Cell 40]
# Optimize the classification threshold using the validation set
# The test set will remain untouched until the final evaluation.

from sklearn.metrics import f1_score

# Get malignant probabilities for the validation set
val_prob = ann_model.predict(X_val, verbose=0).ravel()

thresholds = np.arange(0.10, 0.91, 0.01)

threshold_results = []

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)
    
    threshold_results.append({
        "threshold": threshold,
        "f1": f1_score(
            y_val,
            val_pred,
            zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])
best_val_f1 = float(best_row["f1"])

print("Best validation threshold:", round(best_threshold, 2))
print("Validation F1-score at this threshold:", round(best_val_f1, 4))

# %% [Cell 41]
# Evaluate the baseline ANN using the validation-selected threshold

tuned_test_pred = (
    y_test_prob >= best_threshold
).astype(int)

tuned_accuracy = accuracy_score(
    y_test,
    tuned_test_pred
)

tuned_precision = precision_score(
    y_test,
    tuned_test_pred,
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    tuned_test_pred,
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    tuned_test_pred,
    zero_division=0
)

tuned_auc = roc_auc_score(
    y_test,
    y_test_prob
)

print("THRESHOLD-TUNED BASELINE ANN — TEST RESULTS")
print("=" * 50)
print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy : {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall   : {tuned_recall:.4f}")
print(f"F1-score : {tuned_f1:.4f}")
print(f"ROC-AUC  : {tuned_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tuned_test_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        tuned_test_pred,
        target_names=["Non-Malignant", "Malignant"],
        zero_division=0
    )
)

# %% [Cell 42]
# Final comparison of the three evaluation approaches

final_comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Baseline ANN (0.50)": [
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        test_auc
    ],
    "Improved ANN (0.50)": [
        improved_accuracy,
        improved_precision,
        improved_recall,
        improved_f1,
        improved_auc
    ],
    "Baseline ANN (0.47)": [
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1,
        tuned_auc
    ]
})

print(final_comparison.round(4))

# %% [Cell 43]
# Save the final ANN model and evaluation evidence

import json
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/module5_ann_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save the baseline ANN
ann_model.save(
    OUTPUT_DIR / "baseline_ann_cbis_ddsm.keras"
)

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(
    OUTPUT_DIR / "baseline_training_history.csv",
    index=False
)

# Save baseline confusion matrix
baseline_cm = confusion_matrix(
    y_test,
    y_test_pred
)

pd.DataFrame(
    baseline_cm,
    index=["Actual Non-Malignant", "Actual Malignant"],
    columns=["Predicted Non-Malignant", "Predicted Malignant"]
).to_csv(
    OUTPUT_DIR / "baseline_confusion_matrix.csv"
)

# Save baseline classification report
with open(
    OUTPUT_DIR / "baseline_classification_report.txt",
    "w"
) as f:
    f.write(
        classification_report(
            y_test,
            y_test_pred,
            target_names=["Non-Malignant", "Malignant"],
            zero_division=0
        )
    )

# Save all main metrics
metrics = {
    "accuracy": float(test_accuracy),
    "precision": float(test_precision),
    "recall": float(test_recall),
    "f1_score": float(test_f1),
    "roc_auc": float(test_auc),
    "classification_threshold": 0.50,
    "test_images": int(len(y_test)),
    "training_images": int(len(y_train_split)),
    "validation_images": int(len(y_val)),
    "image_size": "96x96 grayscale"
}

with open(
    OUTPUT_DIR / "baseline_metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=4)

print("Results saved successfully.")
print("\nSaved files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

# %% [Cell 44]
# Create project documentation

documentation = f"""
MODULE 5 — CANCER DIAGNOSIS USING AN ARTIFICIAL NEURAL NETWORK
Dataset: CBIS-DDSM Mammography Dataset

1. DATASET
The CBIS-DDSM dataset was used for binary mammographic cancer classification.
The supplied training dataset contained 1,318 images and the supplied test
dataset contained 378 images.

Pathology labels were mapped as follows:
MALIGNANT = 1
BENIGN = 0
BENIGN_WITHOUT_CALLBACK = 0

2. PREPROCESSING
- Full mammogram JPEG images were used.
- Images were converted to grayscale.
- Images were resized to 96 x 96 pixels.
- Pixel values were normalized from 0–255 to 0–1.
- Missing metadata values were investigated.
- The supplied training and test datasets were kept separate.
- A patient-level 80/20 training-validation split was created.
- No patients appeared in both training and validation sets.

3. BASELINE ANN ARCHITECTURE
Input: 96 x 96 x 1 grayscale image
Flatten
Dense: 256 neurons, ReLU
Dropout: 0.40
Dense: 128 neurons, ReLU
Dropout: 0.30
Dense: 64 neurons, ReLU
Dropout: 0.20
Output: 1 neuron, sigmoid

Total parameters: 2,400,769

4. TRAINING
Optimizer: Adam
Initial learning rate: 0.001
Loss: Binary Cross-Entropy
Batch size: 32
Maximum epochs: 40

EarlyStopping:
- Monitor: validation loss
- Patience: 8
- Restore best weights: True

ReduceLROnPlateau:
- Monitor: validation loss
- Factor: 0.5
- Patience: 3
- Minimum learning rate: 0.000001
- Minimum delta: 0.001

The baseline model completed 11 epochs and restored the weights from
epoch 3.

5. BASELINE TEST RESULTS — THRESHOLD 0.50
Accuracy: {test_accuracy:.4f}
Precision: {test_precision:.4f}
Recall: {test_recall:.4f}
F1-score: {test_f1:.4f}
ROC-AUC: {test_auc:.4f}

Confusion matrix:
[[37, 194],
 [19, 128]]

6. IMPROVEMENT EXPERIMENT
A second ANN was developed using:
- Smaller dense layers: 128, 64, 32
- Batch normalization
- L2 regularization
- Dropout

The improved ANN contained 1,190,913 parameters.

It completed 8 epochs and restored the weights from epoch 1.

Improved ANN test results:
Accuracy: {improved_accuracy:.4f}
Precision: {improved_precision:.4f}
Recall: {improved_recall:.4f}
F1-score: {improved_f1:.4f}
ROC-AUC: {improved_auc:.4f}

The improvement experiment increased accuracy slightly but reduced recall,
F1-score, and ROC-AUC. Therefore, it was not treated as an overall
performance improvement.

7. THRESHOLD EXPERIMENT
A classification threshold was selected using the validation set rather
than the test set.

Validation-selected threshold: {best_threshold:.2f}
Validation F1-score: {best_val_f1:.4f}

When applied to the untouched test set:
Accuracy: {tuned_accuracy:.4f}
Precision: {tuned_precision:.4f}
Recall: {tuned_recall:.4f}
F1-score: {tuned_f1:.4f}
ROC-AUC: {tuned_auc:.4f}

The lower threshold increased malignant recall but substantially increased
false positives and did not improve test F1-score.

8. CRITICAL FINDINGS
The baseline ANN detected many malignant cases, with malignant recall of
87.07%, but it also produced a large number of false-positive predictions.
The overall accuracy and ROC-AUC indicate that the model has limited
discriminative performance in its current form.

The results demonstrate the importance of evaluating multiple metrics rather
than relying on accuracy alone. In a cancer-diagnosis context, false
negatives and false positives have different consequences and should be
considered when interpreting model performance.

The ANN should therefore be treated as an experimental decision-support
model rather than a clinically validated diagnostic system.
"""

documentation_path = OUTPUT_DIR / "project_documentation.txt"

with open(documentation_path, "w") as f:
    f.write(documentation)

print("Project documentation saved successfully.")
print(documentation_path)

# %% [Cell 45]
# Step 40: Identify potential image outliers

def calculate_image_statistics(images):
    stats = []

    for image in images:
        stats.append({
            "mean": float(image.mean()),
            "std": float(image.std()),
            "min": float(image.min()),
            "max": float(image.max())
        })

    return pd.DataFrame(stats)


train_image_stats = calculate_image_statistics(X_train)
test_image_stats = calculate_image_statistics(X_test)


def iqr_outlier_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper


# Identify potential outliers using mean pixel intensity
mean_lower, mean_upper = iqr_outlier_bounds(
    train_image_stats["mean"]
)

train_mean_outliers = (
    (train_image_stats["mean"] < mean_lower) |
    (train_image_stats["mean"] > mean_upper)
)

# Identify potential outliers using pixel standard deviation
std_lower, std_upper = iqr_outlier_bounds(
    train_image_stats["std"]
)

train_std_outliers = (
    (train_image_stats["std"] < std_lower) |
    (train_image_stats["std"] > std_upper)
)


print("TRAINING IMAGE QUALITY CHECK")
print("=" * 40)

print("Training images:", len(train_image_stats))

print(
    "Potential mean-intensity outliers:",
    train_mean_outliers.sum()
)

print(
    "Potential pixel-variation outliers:",
    train_std_outliers.sum()
)

print("\nPixel statistics:")
print(train_image_stats.describe().round(4))

# %% [Cell 46]
# Step 41: Visually inspect potential image outliers

import matplotlib.pyplot as plt
import numpy as np

# Combine the two types of potential outliers
outlier_indices = sorted(
    set(np.where(train_mean_outliers)[0]).union(
        set(np.where(train_std_outliers)[0])
    )
)

print("Potential outlier images to inspect:", len(outlier_indices))
print("Indices:", outlier_indices)

# Prepare labels safely
y_train_array = np.asarray(y_train).reshape(-1)

# Plot the potential outliers
n_images = len(outlier_indices)
n_cols = 4
n_rows = int(np.ceil(n_images / n_cols))

plt.figure(figsize=(14, 3.5 * n_rows))

for plot_num, idx in enumerate(outlier_indices, 1):
    plt.subplot(n_rows, n_cols, plot_num)

    image = X_train[idx]

    # Handle grayscale images with or without a channel dimension
    if image.ndim == 3 and image.shape[-1] == 1:
        image_to_show = image.squeeze()
    else:
        image_to_show = image

    plt.imshow(image_to_show, cmap="gray")
    
    label = int(y_train_array[idx])
    label_name = "Malignant" if label == 1 else "Non-Malignant"

    plt.title(
        f"Index: {idx}\n"
        f"Label: {label_name}\n"
        f"Mean: {train_image_stats.iloc[idx]['mean']:.3f}, "
        f"Std: {train_image_stats.iloc[idx]['std']:.3f}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

# %% [Cell 47]
# Step 42: Check potential outliers for duplicate images

import hashlib
import numpy as np
from itertools import combinations

def image_hash(image):
    """Create a hash for the exact pixel values of an image."""
    return hashlib.sha256(
        np.asarray(image).tobytes()
    ).hexdigest()

# Calculate hashes for the potential outlier images
outlier_hashes = {}

for idx in outlier_indices:
    outlier_hashes[idx] = image_hash(X_train[idx])

# Find exact duplicate groups
hash_groups = {}

for idx, h in outlier_hashes.items():
    hash_groups.setdefault(h, []).append(idx)

duplicate_groups = [
    indices for indices in hash_groups.values()
    if len(indices) > 1
]

print("OUTLIER DUPLICATE CHECK")
print("=" * 40)
print("Potential outlier images:", len(outlier_indices))
print("Exact duplicate groups:", len(duplicate_groups))

if duplicate_groups:
    print("\nExact duplicate groups found:")
    for group in duplicate_groups:
        print(group)
else:
    print("\nNo exact duplicate images found among the potential outliers.")

# Also calculate pixel differences between the similar-looking images
print("\nPAIRWISE DIFFERENCES BETWEEN OUTLIERS")
print("=" * 40)

for idx1, idx2 in combinations(outlier_indices, 2):
    difference = np.mean(
        np.abs(
            X_train[idx1].astype(np.float32) -
            X_train[idx2].astype(np.float32)
        )
    )

    # Show only very similar pairs
    if difference < 0.02:
        print(
            f"Images {idx1} and {idx2}: "
            f"mean absolute difference = {difference:.6f}"
        )

# %% [Cell 48]
# Step 43: Investigate the exact duplicate images

duplicate_indices = [716, 717, 718, 719, 720, 721]

print("DUPLICATE IMAGE INVESTIGATION")
print("=" * 45)

# Check labels
y_train_array = np.asarray(y_train).reshape(-1)

print("\nLabels for duplicate images:")
for idx in duplicate_indices:
    label = int(y_train_array[idx])
    label_name = "Malignant" if label == 1 else "Non-Malignant"
    print(f"Index {idx}: {label_name} (label={label})")

# Check whether all duplicate arrays are exactly identical
print("\nPixel-level comparison:")
reference = X_train[duplicate_indices[0]]

for idx in duplicate_indices[1:]:
    identical = np.array_equal(reference, X_train[idx])
    print(f"Index {duplicate_indices[0]} vs {idx}: {identical}")

# Look for matching rows in the main training dataframe if available
print("\nChecking available dataframe records...")

possible_dataframes = []

for name, obj in globals().items():
    if isinstance(obj, pd.DataFrame):
        if len(obj) >= 1000:
            possible_dataframes.append((name, obj))

for name, df in possible_dataframes:
    relevant_columns = [
        col for col in df.columns
        if any(term in str(col).lower()
               for term in ["image", "path", "patient", "study", "series", "file"])
    ]

    if relevant_columns:
        print(f"\nDataFrame: {name}")
        print("Rows:", len(df))
        print("Relevant columns:", relevant_columns[:15])

        # Display a small sample so we can identify the appropriate source dataframe
        print(df[relevant_columns[:10]].head(3).to_string(index=False))

# %% [Cell 49]
# Step 44: Identify the source records for the duplicate images

duplicate_indices = [716, 717, 718, 719, 720, 721]

print("SOURCE RECORD CHECK")
print("=" * 45)

# Safely take a snapshot of the current global variables
global_items = list(globals().items())

# Find DataFrames with enough rows
dataframe_candidates = []

for name, obj in global_items:
    if isinstance(obj, pd.DataFrame) and len(obj) >= 1000:
        dataframe_candidates.append((name, obj))

print("Candidate DataFrames:")

for name, df in dataframe_candidates:
    relevant_columns = [
        col for col in df.columns
        if any(
            term in str(col).lower()
            for term in [
                "image",
                "path",
                "patient",
                "study",
                "series",
                "file",
                "mass"
            ]
        )
    ]

    print(f"\n{name}")
    print("Rows:", len(df))
    print("Relevant columns:", relevant_columns)

# %% [Cell 50]
# Step 45: Identify the train-index mapping used for X_train

duplicate_indices = [716, 717, 718, 719, 720, 721]

print("TRAIN INDEX MAPPING CHECK")
print("=" * 50)

# Look for index arrays/lists that contain 1318 training observations
global_items = list(globals().items())

mapping_candidates = []

for name, obj in global_items:
    try:
        length = len(obj)
    except TypeError:
        continue

    if length == len(X_train):
        if isinstance(obj, (list, tuple, np.ndarray, pd.Series, pd.Index)):
            mapping_candidates.append(name)

print("Objects with the same length as X_train:", len(mapping_candidates))

for name in mapping_candidates:
    print(" -", name)

print("\nChecking common index variable names:")

for name in [
    "train_idx",
    "train_indices",
    "train_index",
    "train_indices_final",
    "X_train_indices",
    "train_ids",
    "idx_train"
]:
    if name in globals():
        obj = globals()[name]
        print(f"{name}: type={type(obj).__name__}, length={len(obj)}")

print("\nX_train shape:", X_train.shape)
print("y_train shape:", np.asarray(y_train).shape)
print("mass_train shape:", mass_train.shape)

# %% [Cell 51]
# Step 46: Trace duplicate images to their original mass_train records

duplicate_indices = [716, 717, 718, 719, 720, 721]

print("ORIGINAL SOURCE RECORDS FOR DUPLICATES")
print("=" * 60)

# Display the original records corresponding to the duplicate image indices
duplicate_records = mass_train.iloc[duplicate_indices].copy()

columns_to_show = [
    col for col in [
        "patient_id",
        "image view",
        "mass shape",
        "mass margins",
        "pathology",
        "image file path",
        "cropped image file path",
        "ROI mask file path",
        "full_image_path",
        "jpeg_image_path",
        "series_uid",
        "local_image_path"
    ]
    if col in duplicate_records.columns
]

print(duplicate_records[columns_to_show].to_string(index=True))

print("\nNumber of unique values:")
for col in columns_to_show:
    print(f"{col}: {duplicate_records[col].nunique(dropna=False)} unique value(s)")

# %% [Cell 52]
# Step 47: Identify the image paths used to create X_train

print("IMAGE PATH VARIABLE CHECK")
print("=" * 55)

# Look for common image-path variables currently stored in the notebook
possible_path_names = [
    "image_paths",
    "img_paths",
    "paths",
    "all_image_paths",
    "train_image_paths",
    "X_paths",
    "x_paths",
    "jpeg_paths",
    "full_image_paths",
    "local_image_paths",
    "image_file_paths",
    "cropped_image_paths"
]

found_paths = []

for name in possible_path_names:
    if name in globals():
        obj = globals()[name]
        try:
            length = len(obj)
        except TypeError:
            continue

        print(f"\n{name}")
        print("Type:", type(obj).__name__)
        print("Length:", length)

        if length > 0:
            try:
                print("First item:", obj[0])
            except Exception:
                pass

        found_paths.append(name)

print("\nPotential path variables found:", found_paths)

# Check whether mass_train itself contains duplicate full-image references
print("\nDUPLICATE FULL-IMAGE PATH CHECK")
print("=" * 55)

if "jpeg_image_path" in mass_train.columns:
    duplicate_jpeg = mass_train["jpeg_image_path"].duplicated(keep=False)
    print(
        "Rows sharing a JPEG image path:",
        duplicate_jpeg.sum()
    )

    print("\nMost repeated JPEG image paths:")
    print(
        mass_train.loc[duplicate_jpeg, "jpeg_image_path"]
        .value_counts()
        .head(10)
    )

# %% [Cell 53]
# Step 48: Verify the actual source image for the duplicate records

import os
import cv2
import numpy as np

duplicate_indices = [716, 717, 718, 719, 720, 721]

print("VERIFYING ACTUAL SOURCE IMAGES")
print("=" * 60)

for idx in duplicate_indices:
    print(f"\n--- Index {idx} ---")

    # Get the source paths from mass_train
    row = mass_train.iloc[idx]

    for col in ["local_image_path", "jpeg_image_path", "full_image_path"]:
        if col in mass_train.columns:
            path = str(row[col])
            print(f"{col}: {path}")
            print("Exists:", os.path.exists(path))

    # Compare X_train image statistics with the source JPEG if available
    source_path = str(row["local_image_path"])

    if os.path.exists(source_path):
        source_img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)

        if source_img is not None:
            source_resized = cv2.resize(
                source_img,
                (X_train.shape[2], X_train.shape[1])
            ).astype(np.float32) / 255.0

            x_image = np.asarray(X_train[idx]).astype(np.float32)

            # Handle possible channel dimension
            if x_image.ndim == 3:
                x_image = x_image.squeeze()

            difference = np.mean(
                np.abs(source_resized - x_image)
            )

            print("Source image loaded successfully.")
            print("Source image shape:", source_img.shape)
            print("X_train image shape:", x_image.shape)
            print("Mean absolute difference:", difference)

            if difference < 1e-6:
                print("MATCH: X_train image matches the source JPEG.")
            else:
                print("NO EXACT MATCH: preprocessing differs from direct JPEG resize.")
        else:
            print("Could not read the source JPEG.")
    else:
        print("local_image_path does not exist in this environment.")

# %% [Cell 54]
# Step 49: Inspect image preprocessing functions used in the notebook

import inspect

print("IMAGE PREPROCESSING FUNCTIONS")
print("=" * 60)

# Look for functions defined in the notebook that may process images
global_items = list(globals().items())

function_candidates = []

for name, obj in global_items:
    if callable(obj) and (
        "image" in name.lower()
        or "img" in name.lower()
        or "process" in name.lower()
        or "load" in name.lower()
        or "resize" in name.lower()
    ):
        function_candidates.append((name, obj))

print("Potential image-related functions found:")
for name, obj in function_candidates:
    print("-", name)

print("\nFunction source code:")
print("=" * 60)

for name, obj in function_candidates:
    try:
        print(f"\n### {name} ###")
        print(inspect.getsource(obj))
    except Exception:
        print(f"\n### {name} ###")
        print("Source code could not be displayed for this function.")

# %% [Cell 55]
# Step 50: Audit all duplicate source JPEG images in the training data

print("FULL TRAINING DUPLICATE AUDIT")
print("=" * 60)

# Count repeated JPEG image paths
jpeg_counts = mass_train["jpeg_image_path"].value_counts()

# Keep only paths appearing more than once
repeated_jpegs = jpeg_counts[jpeg_counts > 1]

print("Total training records:", len(mass_train))
print("Unique JPEG image paths:", mass_train["jpeg_image_path"].nunique())
print("Records belonging to repeated JPEG paths:", repeated_jpegs.sum())
print("Number of repeated JPEG paths:", len(repeated_jpegs))

print("\nFrequency of repeated JPEG paths:")
print(repeated_jpegs.value_counts().sort_index())

print("\nTop repeated JPEG paths:")
print(repeated_jpegs.head(20))

# %% [Cell 56]
# Step 51: Audit repeated JPEG images by patient, pathology, and ROI

repeated_paths = repeated_jpegs.index

duplicate_audit = (
    mass_train[mass_train["jpeg_image_path"].isin(repeated_paths)]
    .groupby("jpeg_image_path")
    .agg(
        records=("jpeg_image_path", "size"),
        unique_patients=("patient_id", "nunique"),
        unique_pathology=("pathology", "nunique"),
        pathology_values=("pathology", lambda x: ", ".join(sorted(x.astype(str).unique()))),
        unique_cropped_images=("cropped image file path", "nunique"),
        unique_roi_masks=("ROI mask file path", "nunique")
    )
    .sort_values(["records", "unique_pathology"], ascending=[False, False])
)

print("REPEATED JPEG GROUP AUDIT")
print("=" * 70)
print(duplicate_audit.to_string())

print("\nSummary:")
print("Repeated JPEG groups:", len(duplicate_audit))
print("Groups with more than one pathology label:",
      (duplicate_audit["unique_pathology"] > 1).sum())
print("Groups with more than one cropped image:",
      (duplicate_audit["unique_cropped_images"] > 1).sum())
print("Groups with more than one ROI mask:",
      (duplicate_audit["unique_roi_masks"] > 1).sum())

# %% [Cell 57]
# set correct image path for image types
imdir = '../input/cbis-ddsm-breast-cancer-image-dataset/jpeg'

# %% [Cell 58]
# change directory path of images
cropped_images = cropped_images.replace('CBIS-DDSM/jpeg', imdir, regex=True)
full_mammo = full_mammo.replace('CBIS-DDSM/jpeg', imdir, regex=True)
roi_img = roi_img.replace('CBIS-DDSM/jpeg', imdir, regex=True)

# view new paths
print('Cropped Images paths:\n')
print(cropped_images.iloc[0])
print('Full mammo Images paths:\n')
print(full_mammo.iloc[0])
print('ROI Mask Images paths:\n')
print(roi_img.iloc[0])

# %% [Cell 59]
# organize image paths
full_mammo_dict = dict()
cropped_images_dict = dict()
roi_img_dict = dict()

for dicom in full_mammo:
    key = dicom.split("/")[4]
    full_mammo_dict[key] = dicom
for dicom in cropped_images:
    key = dicom.split("/")[4]
    cropped_images_dict[key] = dicom
for dicom in roi_img:
    key = dicom.split("/")[4]
    roi_img[key] = dicom

# view keys
next(iter((full_mammo_dict.items())))

# %% [Cell 60]
print("Example full mammogram paths:")
print(full_mammo.head(5).tolist())

print("\nExample cropped image paths:")
print(cropped_images.head(5).tolist())

print("\nExample ROI image paths:")
print(roi_img.head(5).tolist())

# %% [Cell 61]
# load the mass dataset
mass_train = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv')
mass_test = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv')

mass_train.head()

# %% [Cell 62]
print("TRAINING DATASET SHAPE:")
print(mass_train.shape)

print("\nTEST DATASET SHAPE:")
print(mass_test.shape)

print("\nPATHOLOGY DISTRIBUTION - TRAINING SET:")
print(mass_train["pathology"].value_counts())

print("\nPATHOLOGY PERCENTAGES - TRAINING SET:")
print(mass_train["pathology"].value_counts(normalize=True) * 100)

print("\nMISSING VALUES - TRAINING SET:")
print(mass_train.isnull().sum())

# %% [Cell 63]
print("MISSING VALUES - TEST SET:")
print(mass_test.isnull().sum())

print("\nPATHOLOGY DISTRIBUTION - TEST SET:")
print(mass_test["pathology"].value_counts())

print("\nPATHOLOGY PERCENTAGES - TEST SET:")
print(mass_test["pathology"].value_counts(normalize=True) * 100)

# %% [Cell 64]
# fix image paths
def fix_image_path(data):
    """correct dicom paths to correct image paths"""
    for index, img in enumerate(data.values):
        img_name = img[11].split("/")[2]
        data.iloc[index,11] = full_mammo_dict[img_name]
        img_name = img[12].split("/")[2]
        data.iloc[index,12] = cropped_images_dict[img_name]
        
# apply to datasets
fix_image_path(mass_train)
fix_image_path(mass_test)

# %% [Cell 65]
# check unique values in pathology column
mass_train.pathology.unique()

# %% [Cell 66]
mass_train.info()

# %% [Cell 67]
# rename columns
mass_train = mass_train.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

mass_train.head(5)

# %% [Cell 68]
# check for null values
mass_train.isnull().sum()

# %% [Cell 69]
# fill in missing values using the backwards fill method
mass_train['mass_shape'] = mass_train['mass_shape'].fillna(method='bfill')
mass_train['mass_margins'] = mass_train['mass_margins'].fillna(method='bfill')

#check null values
mass_train.isnull().sum()

# %% [Cell 70]
# quantitative summary of features
mass_train.describe()

# %% [Cell 71]
# view mass_test
mass_test.head()

# %% [Cell 72]
# check datasets shape
print(f'Shape of mass_train: {mass_train.shape}')
print(f'Shape of mass_test: {mass_test.shape}')

# %% [Cell 73]
mass_test.isnull().sum()

# %% [Cell 74]
# check for column names in mass_test
print(mass_test.columns)
print('\n')
# rename columns
mass_test = mass_test.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

# view renamed columns
mass_test.columns

# %% [Cell 75]
# fill in missing values using the backwards fill method
mass_test['mass_margins'] = mass_test['mass_margins'].fillna(method='bfill')

#check null values
mass_test.isnull().sum()

# %% [Cell 76]
# Display some images
import matplotlib.image as mpimg

# create function to display images
def display_images(column, number):
    """displays images in dataset"""
    # create figure and axes
    number_to_visualize = number
    rows = 1
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    
    # Loop through rows and display images
    for index, row in mass_train.head(number_to_visualize).iterrows():
        image_path = row[column]
        image = mpimg.imread(image_path)
        ax = axes[index]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"{row['pathology']}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
print('Full Mammograms:\n')
display_images('image_file_path', 5)
print('Cropped Mammograms:\n')
display_images('cropped_image_file_path', 5)

# %% [Cell 77]
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def image_processor(image_path, target_size):
    """Preprocess images for CNN model"""
    absolute_image_path = os.path.abspath(image_path)
    image = cv2.imread(absolute_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    image_array = image / 255.0
    return image_array

# Merge datasets
full_mass = pd.concat([mass_train, mass_test], axis=0)

# Define the target size
target_size = (224, 224, 3)

# Apply preprocessor to train data
full_mass['processed_images'] = full_mass['image_file_path'].apply(lambda x: image_processor(x, target_size))

# Create a binary mapper
class_mapper = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0} 

# Convert the processed_images column to an array
X_resized = np.array(full_mass['processed_images'].tolist())

# Apply class mapper to pathology column
full_mass['labels'] = full_mass['pathology'].replace(class_mapper)

# Check the number of classes
num_classes = len(full_mass['labels'].unique())

# Split data into train, test, and validation sets (70, 20, 10)
X_train, X_temp, y_train, y_temp = train_test_split(X_resized, full_mass['labels'].values, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Convert integer labels to one-hot encoded labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_val = to_categorical(y_val, num_classes)

# %% [Cell 78]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# %% [Cell 79]
# Define the ANN model
image_size = 64
ann_model = Sequential([
    Flatten(input_shape=(image_size, image_size, 3)),  # Flatten layer to convert 2D image data to 1D
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons and ReLU activation
    Dropout(0.5),  # Dropout layer with dropout rate of 0.5 to prevent overfitting
    Dense(64, activation='relu'),  # Fully connected layer with 64 neurons and ReLU activation
    Dropout(0.5),  # Dropout layer with dropout rate of 0.5
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# %% [Cell 80]
# Compile the model
ann_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# %% [Cell 81]
# Print model summary
ann_model.summary()

# %% [Cell 82]
import tensorflow as tf
from tensorflow.keras import layers, models

# Define your neural network architecture
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),  # Input layer
        layers.Dense(128, activation='relu'),  # Hidden layer
        layers.Dense(10, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Example data (replace this with your actual data loading code)
(train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
val_images = val_images / 255.0

# Hyperparameters
epochs = 10
batch_size = 32

# Create and train the model
try:
    ann_model = create_model()
    ann_history = ann_model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels))
except Exception as e:
    print("An error occurred during training:", e)

# %% [Cell 83]
# Evaluate the model on the validation data
evaluation = ann_model.evaluate(val_images, val_labels)

# Print the evaluation results
print("Evaluation Loss:", evaluation[0])
print("Evaluation Accuracy:", evaluation[1])

# %% [Cell 84]

