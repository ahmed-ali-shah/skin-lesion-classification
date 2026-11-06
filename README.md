# HAM10000 Skin Lesion Classification

**Seven-class skin lesion classification using traditional machine learning and deep learning**

**Course:** F20DL – Data Mining and Machine Learning  
**Academic Year:** 2026/27  
**GitHub Repository:** https://github.com/ahmed-ali-shah/skin-lesion-classification  
**Group:** 1
**Group Members:** Yaaseen Malani H00442735
                   Ahmed Shah H00487992

---

## 1. Project Overview

This project investigates how effectively metadata-based machine-learning models and image-based neural networks can classify the seven diagnostic categories in the HAM10000 skin lesion dataset, particularly under substantial class imbalance.

The project compares traditional machine-learning approaches using patient and lesion metadata against neural-network approaches that learn directly from dermatoscopic images.

The main objective is to evaluate the strengths and limitations of both approaches using appropriate classification metrics, with particular attention to macro F1-score and balanced accuracy.

**Research question:**

How effectively can metadata-based machine-learning models and image-based neural networks classify the seven HAM10000 lesion categories, particularly under substantial class imbalance?

## 2. Dataset

The project uses the HAM10000 dataset, containing 10,015 dermatoscopic images across seven diagnostic categories.

The diagnostic classes are:

| Code | Diagnostic category |
|---|---|
| akiec | Actinic keratoses and intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

The dataset is highly imbalanced, with melanocytic nevi (`nv`) representing the majority of images.

Metadata used in the metadata-based models includes:

- Age
- Sex
- Localization

Identifiers such as `lesion_id` and `image_id` are used for identification and data splitting rather than as model predictors. Diagnosis-related information that could cause target leakage is excluded from the predictors.

**Dataset source:** HAM10000 – A Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions.

**Dataset paper:** Tschandl, P., Rosendahl, C. and Kittler, H. (2018), *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*, Scientific Data.

The dataset is not included in this repository. Users must obtain it from the official dataset source and comply with its licence and usage conditions.

## 3. Methodology

The project is implemented in a Google Colab Jupyter notebook.

### 3.1 Data preparation and exploratory analysis

The workflow includes:

- Loading and inspecting the HAM10000 metadata.
- Examining class distributions and class imbalance.
- Investigating missing values.
- Preprocessing numerical and categorical metadata.
- Exploring relationships between metadata features and diagnostic classes.
- Applying PCA for dimensionality reduction.
- Applying K-Means and Gaussian Mixture Model clustering.

### 3.2 Data splitting

The dataset is divided into training, validation and testing subsets using `StratifiedGroupKFold`, with `lesion_id` used as the grouping variable.

This approach is intended to prevent images belonging to the same lesion from appearing in multiple subsets.

The recorded split is:

| Subset | Number of images |
|---|---:|
| Training | 7,169 |
| Validation | 1,424 |
| Testing | 1,422 |
| **Total** | **10,015** |

The notebook includes a check for lesion overlap between the subsets.

### 3.3 Traditional machine learning

The metadata-based classification experiments include:

- Logistic Regression
- Random Forest
- k-Nearest Neighbours (kNN)
- Decision Tree

The models are evaluated and compared using validation metrics. The selected kNN configuration uses 11 neighbours with distance weighting.

### 3.4 Neural networks and image classification

The image-based experiments include:

- A Multi-Layer Perceptron (MLP).
- A custom Convolutional Neural Network (CNN).
- ResNet18 transfer learning with a frozen backbone.
- Fine-tuning of ResNet18.

The neural-network experiments use image preprocessing and training strategies designed to address class imbalance and improve generalisation.

## 4. Final Test Results

The final held-out test evaluation compares the selected metadata-based kNN model with the fine-tuned ResNet18 image classifier.

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| kNN (Metadata) | 0.6238 | 0.2126 | 0.2057 |
| Fine-tuned ResNet18 (Images) | 0.6181 | 0.6085 | 0.4650 |

The fine-tuned ResNet18 achieves a substantially higher balanced accuracy and macro F1-score than the metadata-based kNN model.

Although kNN obtains slightly higher overall accuracy, its low balanced accuracy and macro F1 indicate poor performance across the minority classes. This demonstrates why accuracy alone is insufficient for evaluating models trained on highly imbalanced datasets.

The results also show that image-based features provide stronger classification performance than the selected metadata-only approach in this experiment.

These results are from the recorded notebook evaluation and are not evidence of clinical diagnostic performance.

## 5. Repository Structure

```text
skin-lesion-classification/
│
├── Copy_of_HAM10000_Data_Preparation.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── results/
    ├── final_test_results.csv
    ├── knn_test_confusion.png
    └── resnet18_test_confusion.png
```

The `results/` files should be added after confirming that the corresponding files exist and match the final notebook evaluation.

## 6. How to Run the Project

The project was developed using Google Colab.

1. Clone or download this repository.
2. Open the notebook in Google Colab.
3. Obtain the HAM10000 dataset from its official source.
4. Upload or mount the dataset in Google Colab.
5. Install the required Python packages.
6. Update the dataset paths if necessary.
7. Execute the notebook cells in order.

**Open in Google Colab:**

[Open the project notebook in Google Colab](https://colab.research.google.com/github/ahmed-ali-shah/skin-lesion-classification/blob/main/Copy_of_HAM10000_Data_Preparation.ipynb)

The notebook includes the data preparation, model training, validation and final evaluation workflow. Reproducing the final results requires the same dataset, split assignments, preprocessing configuration and trained checkpoint used in the recorded evaluation.

## 7. Technologies Used

- Python
- Google Colab
- PyTorch
- Torchvision
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy

## 8. Limitations

- The HAM10000 dataset is highly imbalanced.
- Metadata-only classification has limited ability to distinguish all seven diagnostic categories.
- Some diagnostic categories have relatively few test examples.
- The model's performance on HAM10000 does not establish generalisation to other populations, imaging devices or clinical settings.
- The models are developed for academic research and educational purposes, not for medical diagnosis.

## 9. Disclaimer

This project is an academic machine-learning study. Its models are not medical devices and must not be used to diagnose skin conditions or replace assessment by qualified healthcare professionals.

## 10. Acknowledgements

We acknowledge the authors of the HAM10000 dataset and the researchers who made the dataset available for scientific research.

**Coursework group members:** [Add names and matriculation numbers here if required.]
