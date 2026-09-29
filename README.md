# MD Anderson Cancer Diagnosis Using an Artificial Neural Network

## Project Overview

This project develops an Artificial Neural Network (ANN) for binary breast cancer classification using mammography images from the Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM).

The project was completed as an academic machine-learning decision-making exercise based on the MD Anderson Cancer Institute cancer diagnosis scenario. The model is intended as a research and educational prototype and is not a clinically validated diagnostic system.

## Dataset

The project uses the CBIS-DDSM mammography dataset.

The dataset contains mammography images associated with pathology information for breast abnormalities. The images were prepared for binary classification into:

- Non-Malignant
- Malignant

Dataset source:

Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set]. The Cancer Imaging Archive.

Dataset DOI:

https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

## Data Preprocessing

The image-processing workflow included:

1. Loading the CBIS-DDSM metadata and image paths.
2. Resolving the relevant JPEG image paths.
3. Converting mammograms to grayscale.
4. Resizing images to 96 × 96 pixels.
5. Converting images to NumPy arrays.
6. Normalizing pixel values to the 0–1 range.
7. Checking image statistics for potential visual outliers.
8. Visually inspecting unusual observations.
9. Investigating repeated source-image paths and duplicate observations.

The training data contained 1,318 image records. The data-quality investigation identified repeated JPEG source-image paths and potential visual outliers. These observations were investigated rather than automatically deleted.

## ANN Architecture

The final improved ANN used the following architecture:

96 × 96 Grayscale Image  
↓  
Flatten (9,216 inputs)  
↓  
Dense (128 neurons, ReLU)  
↓  
Batch Normalization  
↓  
Dropout  
↓  
Dense (64 neurons, ReLU)  
↓  
Batch Normalization  
↓  
Dropout  
↓  
Dense (32 neurons, ReLU)  
↓  
Dropout  
↓  
Dense (1 neuron, Sigmoid)  
↓  
Binary Prediction

The model contained:

- Total parameters: 1,190,913
- Trainable parameters: 1,190,529
- Non-trainable parameters: 384

## Training Configuration

The model was implemented using TensorFlow/Keras.

- Optimizer: Adam
- Initial learning rate: 0.001
- Loss function: Binary cross-entropy
- Maximum epochs: 40
- Learning-rate control: ReduceLROnPlateau
- Training control: Early stopping
- Metrics: Binary Accuracy, Precision, Recall, and ROC-AUC

Training stopped at epoch 8, with the best model weights restored from epoch 1.

## Model Evaluation

The final improved ANN produced the following test results:

| Metric | Result |
|---|---:|
| Accuracy | 46.03% |
| Precision | 39.86% |
| Recall | 76.19% |
| F1-score | 52.34% |
| ROC-AUC | 55.30% |

Confusion matrix:

| | Predicted Non-Malignant | Predicted Malignant |
|---|---:|---:|
| Actual Non-Malignant | 62 | 169 |
| Actual Malignant | 35 | 112 |

The results indicate that the model had limited classification performance. Therefore, the ANN should not be interpreted as a clinically reliable autonomous diagnostic system.

## Model Improvement

The baseline ANN was compared with an improved architecture incorporating additional dense layers, batch normalization, dropout, learning-rate reduction, and early stopping.

The improved model increased accuracy from 43.65% to 46.03% and precision from 39.75% to 39.86%. However, recall, F1-score, and ROC-AUC decreased.

This demonstrates that increasing model complexity does not necessarily improve every performance metric.

A separate classification-threshold experiment was also conducted to examine the trade-off between malignant-case recall and false-positive predictions.

## Key Insights

The project demonstrated several important machine-learning and data-quality considerations:

- Image preprocessing is important before ANN training.
- Statistical outliers are not automatically invalid medical observations.
- Repeated source-image records require investigation before deletion.
- Accuracy alone is insufficient for evaluating a cancer-classification model.
- Classification thresholds can substantially change recall and false-positive behaviour.
- The ANN architecture used in this project does not explicitly capture spatial relationships in mammography images.
- CNNs and transfer-learning approaches could be investigated in future work because they are designed to process spatial image patterns.

## Ethical and Clinical Considerations

This project is an academic prototype and should not be used to make autonomous clinical decisions.

Potential future clinical application would require:

- External validation on independent patient populations
- Patient-level or group-based data splitting
- Evaluation of false positives and false negatives
- Explainability and uncertainty assessment
- Clinical validation
- Appropriate privacy and security controls
- Continuous model-performance monitoring
- Human clinical oversight

The proposed role of the technology is therefore decision support rather than replacement of clinical judgment.

## Future Improvements

Future work could investigate:

1. Convolutional Neural Networks (CNNs)
2. Transfer learning using pretrained image models
3. Data augmentation
4. Patient-level/group-based train-validation-test splitting
5. Additional external datasets
6. Explainable AI techniques
7. More systematic threshold selection
8. Clinical validation and prospective evaluation

## Files

This repository contains the Python source code used for the ANN implementation.

The complete CBIS-DDSM dataset is not included because the dataset is available from its original source.

## Citation

Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., & Rubin, D. L. (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data, 4*, 170177. https://doi.org/10.1038/sdata.2017.177
