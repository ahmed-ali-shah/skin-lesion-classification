# HAM10000 Skin Lesion Classification

**Seven-class skin lesion classification with metadata-based classical models and image-based neural networks**

**Course:** F20DL – Data Mining and Machine Learning
**Academic year:** 2026/27
**Group:** 1

| Member | Matriculation number |
|---|---|
| Ahmed Shah | H00487992 |
| Yaaseen Malani | H00442735 |
| Dhruv Laha | H00490130 |
| Jabez Vas | H00430097 |
| Misha Chandel | H00445699 |

---

## 1. Project overview

The project asks how well the seven HAM10000 lesion classes can be separated using only patient metadata (age, sex and body site), and how much image-based neural networks add, under strong class imbalance. Models are selected on validation **Macro F1**, with balanced accuracy reported alongside.

All experiments use one fixed **lesion-grouped** train / validation / test split (`StratifiedGroupKFold`, grouped by `lesion_id`), so no lesion appears in more than one partition.

| Partition | Images | Lesions |
|---|---:|---:|
| Train | 7,169 | 5,335 |
| Validation | 1,424 | 1,068 |
| Test | 1,422 | 1,067 |

## 2. Dataset

HAM10000 (Tschandl, Rosendahl and Kittler, *Scientific Data*, 2018) contains 10,015 dermoscopic images in seven classes: akiec, bcc, bkl, df, mel, nv and vasc. It is available from [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) under the **CC BY-NC 4.0** licence (non-commercial use, with attribution).

This repository includes the metadata CSV and our split assignments in `data/`. **It does not include the images**: download `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip` from Dataverse.

Predictors for the metadata models: `age`, `sex`, `localization`. The ID columns, `dataset` and `dx_type` are excluded (`dx_type` leaks the label).

## 3. Repository structure

```text
skin-lesion-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── HAM10000_metadata.csv        original metadata (10,015 rows)
│   └── HAM10000_splits.csv          frozen train/val/test assignment used for every result
├── notebooks/
│   ├── 01_data_split_classical_ml_and_cnns.ipynb
│   ├── 02_r3_additional_experiments.ipynb
│   ├── 03_resnet18_extended_finetuning.ipynb
│   └── original/                    the unedited Colab notebooks, with every output, error and recovery cell
│       ├── Copy_of_HAM10000_Data_Preparation.ipynb
│       ├── HAM10000_R3_Outstanding_Experiments.ipynb
│       └── HAM10000_Model_Improvement.ipynb
└── results/
    ├── final_test_results.csv, knn_test_confusion.png, resnet18_test_confusion.png   held-out test
    ├── model_config.json            model choices frozen before testing
    ├── small_cnn_*, best_small_cnn.pth, cnn_model_comparison.csv
    ├── metadata_feature_importance.csv / .png
    └── r3_experiments/              outputs of notebook 02 (Naive Bayes, decision tree,
                                     cross-validation, clustering sweep, CNN ablation, ResNet18 check)
```

The three numbered notebooks are cleaned, ordered versions of the originals: recovery cells (from Colab disconnections) and failed cells are left out, and markdown explains each step. Their outputs are the outputs of the original Colab runs. The originals are kept unedited in `notebooks/original/`.

## 4. How to run

The project was run in **Google Colab with a T4 GPU**.

1. In Google Drive, create this folder:

   ```text
   MyDrive/DMML Project/
   ├── data/
   │   ├── HAM10000_metadata.csv      (copy from data/ in this repository)
   │   └── HAM10000_splits.csv        (copy from data/ in this repository)
   └── HAM10000 Images/
       ├── HAM10000_images_part_1.zip
       └── HAM10000_images_part_2.zip
   ```

2. Open the notebooks in Colab, choose *Runtime → Change runtime type → T4 GPU*, and run them in order:

   | Order | Notebook | What it does |
   |---|---|---|
   | 1 | [01_data_split_classical_ml_and_cnns](https://colab.research.google.com/github/ahmed-ali-shah/skin-lesion-classification/blob/main/notebooks/01_data_split_classical_ml_and_cnns.ipynb) | Split, EDA, LR / RF / kNN / decision tree, metadata MLP, clustering, tuning, feature importance, small CNN, frozen and fine-tuned ResNet18, held-out test |
   | 2 | [02_r3_additional_experiments](https://colab.research.google.com/github/ahmed-ali-shah/skin-lesion-classification/blob/main/notebooks/02_r3_additional_experiments.ipynb) | Naive Bayes, decision-tree grid, grouped 5-fold CV, K-Means sweep, CNN class-weight ablation, checkpoint checks |
   | 3 | [03_resnet18_extended_finetuning](https://colab.research.google.com/github/ahmed-ali-shah/skin-lesion-classification/blob/main/notebooks/03_resnet18_extended_finetuning.ipynb) | Extended fine-tuning of ResNet18 (validation only) |

   Notebook 02, Part 9 re-checks the checkpoint created by notebook 03, so run that part after notebook 03.

3. Checkpoints and results are saved into the same Drive folder. The ResNet18 checkpoints (about 43 MB each) are not stored in this repository.

## 5. Where each report result comes from

| Report item | Notebook and part |
|---|---|
| Split and Table 1 | 01, Part 1 |
| Figure 1 (class counts, age, body site) | 01, Part 2 |
| Clustering (§2.5) | 01, Part 5 · k-sweep: 02, Part 6 |
| Feature importance (§2.5, Appendix D.1) | 01, Part 7 |
| Tuning grids (Table 2) | 01, Part 6 · decision tree: 02, Part 3 · Naive Bayes: 02, Part 2 |
| Table 3 (metadata models, CV) | 01, Parts 4 and 6 · 02, Parts 2–5 |
| Table 4 (image models) | 01, Parts 9–11 · ablation: 02, Part 8 · extended: 03 |
| Table 5 and Figure 3 (held-out test) | 01, Part 12 |

## 6. Results

**Held-out test set (1,422 images), run once after model selection:**

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Always predict NV | 0.688 | 0.143 | 0.116 |
| kNN, k = 11, distance-weighted (metadata) | 0.624 | 0.213 | 0.206 |
| Fine-tuned ResNet18 (images) | 0.618 | 0.609 | 0.465 |

The two models have almost the same accuracy, but only the image model recognises the minority classes.

**Extended fine-tuning (validation only):** ResNet18 reached 0.610 Macro F1, 0.728 accuracy and 0.703 balanced accuracy at epoch 17. It was trained after the test set had been used once, so it has no held-out test score.

## 7. Reproducibility notes

- `data/HAM10000_splits.csv` is the authoritative split. Newer scikit-learn versions shuffle `StratifiedGroupKFold` differently, so re-running the split code does not regenerate the same assignment. Notebook 01 therefore reloads the saved split straight after the split code.
- Random seeds are 123 or 42, as set in each notebook.
- Some summary cells in notebook 02 show values computed in notebook 01 as typed-in numbers; the notebook says where each one comes from.

## 8. Limitations

- HAM10000 is highly imbalanced; DF and VASC have only 14–22 validation and test images each.
- Grouping by lesion does not guarantee patient independence, and the data come from only two clinics.
- The neural-network results come from single runs with one seed.
- The models are for academic study only and are **not** suitable for medical diagnosis.

## 9. Acknowledgements

We thank the authors of the HAM10000 dataset: P. Tschandl, C. Rosendahl and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," *Scientific Data* 5, 180161 (2018).

## 10. AI-use declaration

Generative AI tools were used in this project as follows.

**Code.** AI assistants were used as coding aids while writing the notebooks — for example, to suggest and debug code, explain library errors and help structure parts of the training and evaluation code. The group designed the experiments, chose the models and hyperparameter grids, ran all of the code in Google Colab and checked every reported result against the saved outputs.

**Repository organisation (Claude, Anthropic).** Claude was used to tidy and document the repository before submission:
- reorganised the notebooks into three numbered notebooks (01–03) in run order, built from the group's existing cells, with markdown headings and notes added; the unedited originals are kept in `notebooks/original/`
- changed the first data-loading cell in notebooks 01 and 02 to read the CSV files from Google Drive instead of `files.upload()`, and added one cell in notebook 01 that reloads the saved split file
- rewrote this README and added missing packages to `requirements.txt`

No experiment logic, model settings or results were changed by these edits.
