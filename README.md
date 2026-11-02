# 🧬 IDC Histopathology: Adversarial Robustness & Deep Learning Reliability

![Python](https://img.shields.io/badge/Python-Deep%20Learning-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white)
![CNN](https://img.shields.io/badge/Models-CNN%20%7C%20ResNet18%20%7C%20MobileNetV3-00897B)
![Adversarial ML](https://img.shields.io/badge/Adversarial-Robustness-8E44AD)
![Research](https://img.shields.io/badge/Research-Retrospective%20Patch--Level-455A64)

This project investigates the **adversarial robustness, classification performance, and reliability of deep learning models for invasive ductal carcinoma (IDC) histopathology patch classification**.

Using 277,524 publicly available breast histopathology image patches, the study compares a custom Convolutional Neural Network (CNN), ImageNet-pretrained ResNet-18, and MobileNetV3-Small.

Beyond conventional classification metrics, the analysis examines how adversarial perturbations affect model predictions, particularly the risk of incorrectly classifying IDC-positive patches as negative.

The project integrates adversarial attacks, defensive training, cross-model transferability, explainable AI, probability calibration, selective prediction, and statistical uncertainty analysis.

The primary objective is not simply to maximize classification accuracy, but to understand **when deep learning models fail, how vulnerable they are to adversarial perturbations, and whether defensive techniques improve their reliability**.

The research is a retrospective, patch-level methodological study. It does not represent a clinically validated patient-level cancer diagnosis system.

---

## 🔍 Project Overview

### Objective

- Develop and evaluate deep learning models for binary IDC histopathology patch classification.
- Compare custom CNN, pretrained ResNet-18, and pretrained MobileNetV3-Small architectures.
- Measure classification performance using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Investigate model vulnerability to FGSM, PGD, DeepFool, and AutoAttack-based perturbations.
- Evaluate targeted attacks and cross-model adversarial transferability.
- Compare clean fine-tuning, FGSM adversarial training, and PGD adversarial training.
- Examine false negatives, prediction confidence, calibration, and selective prediction.
- Analyze model behavior using Grad-CAM visualizations.
- Assess statistical uncertainty and sensitivity to alternative data splits.

### Key Questions

1. How accurately can different deep learning architectures classify IDC-positive and IDC-negative histopathology patches?
2. How does adversarial perturbation strength affect classification performance and false-negative errors?
3. Are pretrained architectures more robust than a custom CNN?
4. Can adversarial examples generated against one architecture transfer to another?
5. How effectively do FGSM and PGD adversarial training improve robustness?
6. Can high accuracy against a single attack provide a misleading impression of robustness?
7. How do adversarial attacks change model attention and prediction confidence?
8. How stable are the observed results across alternative data splits and repeated training runs?

### Dataset Overview

| Dataset | Description |
|---|---|
| Data source | Breast Histopathology Images |
| Source repository | Kaggle |
| Image type | H&E-stained breast histopathology patches |
| Original patch size | 50 × 50 pixels |
| Classification target | IDC-positive vs. IDC-negative |
| Total unique patches | 277,524 |
| IDC-negative patches | 198,738 |
| IDC-positive patches | 78,786 |
| Filename-derived groups | 279 |
| Source-reported whole-mount slides | 162 |
| Training patches | 185,782 |
| Validation patches | 40,120 |
| Test patches | 51,622 |

**Dataset:** https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

**Original source documentation:** https://andrewjanowczyk.com/use-case-6-invasive-ductal-carcinoma-idc-segmentation/

The source collection contains image patches extracted from breast tissue slides, with binary labels indicating whether IDC is present within each patch.

Duplicate image paths and overlapping dataset partitions were examined during preprocessing.

The project uses filename-derived group identifiers to separate training, validation, and test records.

**Important:** The 279 parsed groups should not automatically be interpreted as 279 independently verified patients. The source-reported number of slides and the number of parsed filename groups represent different concepts. Full patient-to-slide provenance requires additional source-level verification.

---

## 🧠 Methodology

### 1. Data Preparation & Quality Validation

The dataset was prepared for deep learning through image loading, deduplication checks, label validation, and partition integrity testing.

The workflow includes:

- Loading IDC-positive and IDC-negative image patches.
- Parsing source filenames and extracting group identifiers.
- Checking duplicate file paths and image records.
- Validating binary labels against available filename and directory information.
- Creating group-disjoint training, validation, and test partitions.
- Applying training-only image augmentation.
- Resizing images to match model input requirements.
- Standardizing tensor transformations and preprocessing.
- Checking for class imbalance.
- Preserving split configuration and random seeds for reproducibility.

Group separation helps reduce leakage between the training and evaluation sets, although it does not independently establish patient-level clinical separation.

### 2. Deep Learning Model Development

Three architectures were developed and compared.

| Model | Description |
|---|---|
| Custom CNN | Convolutional neural network trained for binary IDC patch classification |
| ResNet-18 | ImageNet-pretrained residual network adapted for binary classification |
| MobileNetV3-Small | Lightweight ImageNet-pretrained architecture adapted for binary classification |

The modeling workflow includes:

- Class-balanced sampling.
- Training-only data augmentation.
- Transfer learning for pretrained architectures.
- Validation-based model selection.
- Early stopping.
- Checkpoint saving and reloading.
- Classification threshold optimization using validation predictions.
- Independent evaluation on the held-out internal test partition.
- Repeated training and alternative-split experiments.

The project evaluates both conventional classification performance and robustness under intentionally perturbed inputs.

### 3. Adversarial Attack Analysis

Adversarial examples are generated by introducing small perturbations designed to change model predictions.

The analysis includes:

**Fast Gradient Sign Method (FGSM)**

A single-step gradient-based attack that perturbs the input in a direction that increases classification loss.

**Projected Gradient Descent (PGD)**

An iterative gradient-based attack with projection onto an allowed perturbation region.

**DeepFool**

An iterative method that estimates perturbations needed to cross a model's decision boundary.

**AutoAttack — Custom Configuration**

An additional robustness evaluation using APGD-CE and Square Attack.

This configuration is not equivalent to the complete standard AutoAttack ensemble.

**Targeted FGSM and PGD**

Attacks designed to move predictions toward a specified target class, including IDC-negative predictions.

**Margin-PGD**

A stronger gradient-based evaluation using a margin-oriented objective to examine robustness around the model's classification decision boundary.

**Worst-of Attack Evaluation**

For each evaluated sample, multiple attack candidates are considered to identify the most damaging observed perturbation.

The analysis also examines:

- Multiple perturbation magnitudes.
- PGD convergence at 10, 20, and 50 iterations.
- Multiple random restarts.
- Attack effectiveness across different architectures.
- False-negative behavior after perturbation.
- Independent attack implementation checks.
- Perturbation constraints and input validity.

For the primary FGSM and PGD evaluations, the perturbation budget follows an L∞-bounded threat model, with ε values including 2/255, 4/255, and 8/255.

DeepFool is evaluated separately because its perturbation formulation is not directly equivalent to the L∞ threat model used for FGSM and PGD.

### 4. Adversarial Defense & Robustness Evaluation

The project compares multiple approaches to improving robustness.

The defense experiments include:

- Original CNN without adversarial training.
- Clean fine-tuning.
- FGSM adversarial training.
- PGD adversarial training.

Each defense is evaluated using clean inputs and adversarially perturbed inputs.

Particular attention is given to the relationship between clean classification accuracy and robustness under stronger attacks.

Additional evaluation examines potential gradient masking, where a model appears resistant to one attack but remains vulnerable to more effective optimization methods.

### 5. Cross-Model Attack Transferability

Adversarial examples generated against one architecture are evaluated against other trained architectures.

The analysis investigates:

- CNN-to-ResNet transferability.
- CNN-to-MobileNet transferability.
- ResNet-to-CNN transferability.
- ResNet-to-MobileNet transferability.
- MobileNet-to-CNN transferability.
- MobileNet-to-ResNet transferability.

Transferability analysis examines whether model vulnerabilities are architecture-specific or shared across different networks.

### 6. Explainable AI & Model Interpretation

Grad-CAM is used to visualize spatial regions influencing model predictions.

The analysis includes:

- Attention visualizations for clean histopathology images.
- Attention visualizations for adversarial examples.
- Comparison of activation patterns before and after perturbation.
- Inspection of predictions on correctly and incorrectly classified patches.

Grad-CAM provides qualitative information about model activation patterns, but its heatmaps do not independently establish medically valid explanations.

### 7. Prediction Reliability & Statistical Analysis

The project evaluates predictive reliability beyond classification accuracy.

Methods include:

- ROC-AUC and Precision–Recall analysis.
- Confusion matrices.
- False-negative and false-positive analysis.
- Probability calibration.
- Selective prediction and abstention.
- Coverage–performance trade-offs.
- Group-cluster bootstrap confidence intervals.
- Paired clean-versus-attacked comparisons.
- Alternative group split stability experiments.
- Repeated training seed comparisons.
- Sensitivity analysis under simulated staining and image variations.

Bootstrap uncertainty is calculated using the available filename-derived group definitions.

These estimates describe uncertainty within the observed internal dataset and should not be interpreted as external clinical validation.

---

## 📊 Key Insights

### 1. Pretrained Models Outperformed the Custom CNN

The three models were evaluated using 51,622 held-out test patches.

Classification thresholds were selected using validation-set F1-score optimization.

| Model | Accuracy | IDC+ Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Custom CNN | 87.94% | 78.62% | 0.7731 | 0.9374 |
| ResNet-18 | **89.40%** | 85.08% | **0.8075** | **0.9514** |
| MobileNetV3-Small | 89.22% | **86.21%** | 0.8070 | 0.9512 |

ResNet-18 achieved the highest overall accuracy, F1-score, and ROC-AUC.

MobileNetV3-Small achieved the highest IDC-positive recall, indicating that it identified a slightly larger proportion of positive patches at the selected threshold.

The performance differences between the two pretrained architectures were relatively small.

These results describe internal patch-level classification and do not establish superiority across independent institutions or clinical populations.

### 2. Adversarial Attacks Substantially Reduced Classification Performance

Both FGSM and PGD demonstrated that models with strong clean-image classification performance can be highly vulnerable to adversarial perturbations.

In the full-test evaluation, PGD with ε = 8/255 substantially reduced classification accuracy across all three baseline models.

| Model | Clean Accuracy | PGD Accuracy |
|---|---:|---:|
| Custom CNN | 85.35% | 0.90% |
| ResNet-18 | 86.39% | 0.017% |
| MobileNetV3-Small | 88.44% | 0.00% |

These values use the default 0.50 classification threshold and are separate from the validation-F1 threshold results in the baseline performance table.

The findings highlight how adversarial perturbations can undermine model reliability, even when clean-image performance is relatively high.

Because adversarial results depend on the precise threat model and attack implementation, the project includes additional independent attack checks rather than treating a single attack run as sufficient evidence.

### 3. Stronger Attacks Exposed Weaknesses in FGSM-Based Defense

One of the most important findings concerns the difference between FGSM and PGD adversarial training.

A supplementary evaluation using 384 test patches compared clean performance with FGSM and stronger margin-based PGD attacks.

| Defense Model | Clean Accuracy | FGSM Accuracy | Strong PGD Accuracy |
|---|---:|---:|---:|
| FGSM-trained CNN | 86.72% | **99.22%** | **0.00%** |
| PGD-trained CNN | 78.13% | 71.88% | **70.31%** |

The FGSM-trained model achieved very high accuracy against the evaluated FGSM attack but failed under stronger iterative attacks.

This behavior is consistent with possible gradient masking, label leaking, or overfitting to a specific attack procedure.

The result demonstrates why high performance against one adversarial attack should not automatically be interpreted as general robustness.

By comparison, PGD adversarial training maintained substantially higher accuracy under the stronger evaluated attack.

These comparisons describe the observed internal evaluation rather than a universal guarantee of adversarial robustness.

### 4. Robustness Improvements Involved Trade-Offs

Adversarial training influenced both clean-image accuracy and attacked-image performance.

In the supplementary evaluation, the PGD-trained CNN achieved approximately 78.13% clean accuracy while maintaining approximately 70.31% accuracy under the evaluated stronger PGD attack.

The result illustrates a trade-off between clean classification performance and resistance to adversarial perturbations.

A model with the highest clean accuracy is not necessarily the most robust model.

The project therefore compares clean performance, adversarial performance, positive-class recall, and false-negative behavior rather than selecting models using one metric alone.

### 5. Attack Effectiveness Depends on the Evaluation Method

The project uses several complementary robustness checks:

- FGSM and iterative PGD.
- PGD convergence analysis.
- DeepFool.
- Custom AutoAttack.
- Margin-oriented PGD.
- Multiple attack restarts.
- Worst-of candidate selection.

These experiments evaluate attack effectiveness from different perspectives.

In particular, stronger iterative attacks helped identify weaknesses that were not apparent from FGSM evaluation alone.

DeepFool and AutoAttack are evaluated on smaller samples than the primary full-test FGSM and PGD experiments.

Their results should therefore be interpreted alongside their evaluation sample sizes and configurations.

### 6. Cross-Model Transferability Revealed Shared Vulnerabilities

The transferability experiments examine whether adversarial examples created for one architecture also affect other models.

The analysis compares attack success across custom CNN, ResNet-18, and MobileNetV3-Small.

This provides an additional perspective on adversarial robustness by examining whether vulnerabilities are specific to individual networks or transferable across architectures.

Transferability is evaluated on a selected internal test subset, and the results are interpreted as exploratory evidence rather than deployment-level black-box security validation.

### 7. Model Reliability Extends Beyond Accuracy

The project also investigates reliability through probability calibration, selective prediction, and visual explanation.

Selective prediction evaluates how classification performance changes when the model abstains from uncertain predictions.

Grad-CAM is used to compare model activation patterns between clean and perturbed images.

Simulated staining variations and image corruptions provide additional sensitivity checks.

These analyses demonstrate that evaluating a medical-image classification model requires more than conventional accuracy metrics.

However, improved calibration, selective prediction, or visual interpretability does not independently establish clinical safety.

### 8. Alternative Data Splits Provided Additional Stability Evidence

Model performance was also examined using two alternative filename-group-disjoint splits.

| Model | Alternative Splits | Mean Accuracy | Mean IDC+ Recall |
|---|---:|---:|---:|
| Custom CNN | 2 | 89.27% | 80.51% |
| ResNet-18 | 2 | 89.27% | 80.74% |
| MobileNetV3-Small | 2 | 88.82% | **82.00%** |

These experiments provide exploratory information about sensitivity to data partitioning.

Two alternative splits are insufficient to establish external generalizability, and the underlying filename-derived groups still require careful provenance interpretation.

---

## 📈 Data Visualization

The following visualizations summarize the project's main analyses.

The figures are organized to show the progression from dataset preparation and baseline modeling to adversarial attacks, defense evaluation, explainability, and reliability analysis.

### 1. Breast Histopathology Dataset

![Breast Histopathology Dataset](figures/01_Tissue_examples.png)

Examples of IDC-positive and IDC-negative histopathology patches used for model training and evaluation.

The visualization provides an overview of the image characteristics and the binary classification task.

### 2. Model Performance Under Adversarial Attacks

![CNN Adversarial Performance](figures/02_Baseline_overview.png)

Comparison of CNN classification performance on clean images and adversarial examples generated using FGSM and PGD.

The results illustrate how small input perturbations can substantially affect model predictions.

### 3. ResNet-18 Classification Performance

![ResNet-18 Diagnostics](figures/03_ResNet18_diagnostic.png)

Evaluation of ResNet-18 using a confusion matrix, ROC curve, and precision–recall curve.

These metrics provide complementary information about classification performance and positive-class detection.

### 4. Attack Strength and False Negative Risk

![Adversarial Strength and False Negatives](figures/04_Attack_epsilon_vs_FNR.png)

Analysis of how increasing adversarial perturbation strength affects model accuracy and false-negative errors.

The visualization highlights the importance of assessing IDC-positive recall under adversarial conditions.

### 5. PGD Attack Convergence Analysis

![PGD Convergence](figures/05_PGD_convergence.png)

Comparison of attack effectiveness across different PGD iteration counts.

This analysis examines whether increasing attack optimization steps produces more effective adversarial examples.

### 6. Comparing Adversarial Defense Strategies

![Adversarial Defense Comparison](figures/06_Defense_tradeoff.png)

Comparison of clean accuracy and adversarial accuracy across baseline, FGSM-trained, and PGD-trained models.

The results highlight the limitations of FGSM-only defense and the trade-offs associated with stronger adversarial training.

### 7. Cross-Model Attack Transferability

![Adversarial Transferability](figures/07_Transferability_heatmap.png)

Heatmap showing how adversarial examples generated against one architecture affect other deep learning models.

The analysis examines vulnerability patterns shared across CNN, ResNet-18, and MobileNetV3.

### 8. Visualizing Model Attention with Grad-CAM

![Grad-CAM Analysis](figures/08_GradCAM_ResNet18.png)

Comparison of ResNet-18 activation patterns for original and adversarially perturbed histopathology patches.

Grad-CAM provides a qualitative view of model behavior but does not independently validate histopathological reasoning.

### 9. Selective Prediction and Model Reliability

![Selective Prediction](figures/09_Selective_prediction.png)

Analysis of how abstaining from uncertain predictions changes evaluation coverage and classification performance.

The visualization explores the relationship between predictive confidence and observed reliability.

### 10. Model Sensitivity to Image Variations

![Image Shift Sensitivity](figures/10_Stain_corruption_dose_response.png)

Evaluation of model sensitivity to simulated staining variations and image distortions.

These experiments examine robustness to synthetic image changes and do not substitute for testing on independent clinical data.

**Additional analyses**, including calibration, targeted attacks, PGD strength validation, confidence intervals, repeated training, and alternative split experiments, are available in the full Jupyter Notebook and accompanying research outputs.

---

## 🛠 Tech Stack

**Programming Language**
- Python

**Development Environment**
- Jupyter Notebook
- Visual Studio Code

**Deep Learning**
- PyTorch
- Torchvision
- CUDA

**Data Processing**
- Pandas
- NumPy
- Pillow

**Machine Learning & Evaluation**
- Scikit-learn
- SciPy

**Adversarial Machine Learning**
- FGSM
- PGD
- DeepFool
- Custom AutoAttack
- Targeted Adversarial Attacks
- Margin-PGD
- Adversarial Training
- Transferability Analysis

**Explainability & Visualization**
- Grad-CAM
- Matplotlib
- Seaborn

**Statistical Methods**
- Group-Cluster Bootstrap
- Confidence Intervals
- Paired Performance Comparisons
- Model Calibration
- Selective Prediction
- Alternative Split Stability Analysis

**Hardware Used**
- NVIDIA GeForce RTX 4070 (12GB VRAM)
- Intel Core i9-13900K
- 64GB RAM

---

## 🗂️ Repository Structure

The following is a recommended repository layout for publishing the research. Paths should match the files actually uploaded to GitHub.

```text
IDC-Histopathology-Adversarial-Robustness/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── IDC_RESEARCH_FINAL_PROVENANCE_HARDENED.ipynb
│
├── figures/
│   ├── 01_Tissue_examples.png
│   ├── 02_Baseline_overview.png
│   ├── 03_ResNet18_diagnostic.png
│   ├── 04_Attack_epsilon_vs_FNR.png
│   ├── 05_PGD_convergence.png
│   ├── 06_Defense_tradeoff.png
│   ├── 07_Transferability_heatmap.png
│   ├── 08_GradCAM_ResNet18.png
│   ├── 09_Selective_prediction.png
│   └── 10_Stain_corruption_dose_response.png
│
├── research/
│   ├── MANUSCRIPT_AUTHOR_REVIEW.md
│   ├── DATA_SOURCE_AND_IDENTITY_VERIFICATION.md
│   └── AUTHOR_SUBMISSION_CHECKLIST.md
│
├── scripts/
│   └── verify_idc_provenance.py
│
└── outputs/
    └── README.md
```

The `notebooks/` directory contains the full research workflow, including model training, adversarial evaluation, defensive training, statistical analysis, and reporting.

The `figures/` directory contains ten selected visualizations from the executed experiments.

The `research/` directory contains supporting documentation, research limitations, and manuscript review materials.

The `scripts/` directory provides additional data integrity verification utilities.

The `outputs/` directory can document the expected output structure. Large generated checkpoints, temporary attack tensors, and source images do not need to be committed to the repository.

---

## 🚀 How to Run

### 1. Install the Required Libraries

Clone or download the repository.

If the repository includes a dependency file, install the required packages using:

```bash
python -m pip install -r requirements.txt
```

The project uses PyTorch and torchvision for deep learning, along with additional packages for adversarial attacks, model evaluation, and visualization.

For GPU acceleration, install a CUDA-compatible PyTorch build appropriate for the system and NVIDIA driver.

Official installation instructions:

https://pytorch.org/get-started/locally/

### 2. Download the Dataset

Download the Breast Histopathology Images dataset:

https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

Extract the dataset into a local directory.

The source image files are not included in this repository.

### 3. Configure the Dataset Location

Open the main notebook and update the configured dataset path.

For example:

```python
DATA_ROOT = r"I:\Potforlio\IDC Adversarial Robustness"
```

This is the original local workspace path and should be changed to match the user's environment.

Ensure the configured location contains the expected source image folders.

### 4. Configure the Research Stage

The notebook supports multiple execution profiles and stages.

For the complete research configuration:

```python
RUN_PROFILE = "all_research"
```

The main execution stages are:

**Baseline model training**

```python
RUN_STAGE = "baseline_only"
```

**Alternative split stability analysis**

```python
RUN_STAGE = "split_stability"
```

**Adversarial attacks and advanced analyses**

```python
RUN_STAGE = "analyses"
```

The notebook also includes an integrated execution configuration for running the complete pipeline.

Full research execution is computationally expensive, especially for iterative adversarial attacks and repeated training experiments.

### 5. Execute the Notebook

Open:

```text
notebooks/IDC_RESEARCH_FINAL_PROVENANCE_HARDENED.ipynb
```

Run the cells in order.

The workflow saves model checkpoints, evaluation tables, statistical analyses, figures, and research audit outputs to the configured directories.

Previously trained checkpoints can be reused when their configuration and dataset split identifiers match the expected experiment settings.

### 6. Verify Data Integrity

Run the provenance verification utility after configuring the required file paths:

```bash
python scripts/verify_idc_provenance.py
```

The verification workflow checks supported filename, grouping, label, and partition consistency properties.

Automated consistency checks cannot independently establish the biological identity of individual patients or the complete relationship between patients, slides, and image patches.

### 7. Review the Research Outputs

Generated results include model evaluation metrics, attack comparisons, defense performance, statistical uncertainty estimates, and research documentation.

The notebook also produces automated technical audit outputs.

Passing an automated technical audit indicates that expected computational evidence is present; it does not constitute independent scientific review or approval for clinical use.

---

## ⚠️ Limitations

The project has several important limitations.

- **Patch-level evaluation:** Models classify individual histopathology patches rather than diagnosing patients.
- **Single public dataset:** The study uses one publicly available image collection without independent institutional validation.
- **Patient identity verification:** Parsed filename groups are not equivalent to independently confirmed unique patients.
- **Source provenance:** Full patient-to-slide mappings require additional source documentation.
- **Image resolution:** Original 50×50 patches contain limited spatial context compared with whole-slide images.
- **Adversarial threat model:** Results depend on the defined perturbation constraints, attack implementation, and optimization settings.
- **Gradient masking:** Apparent robustness under a weak attack can be misleading, as illustrated by the FGSM-trained model.
- **Evaluation sample sizes:** DeepFool, AutoAttack, and supplementary attack checks use smaller samples than the full-test FGSM/PGD analysis.
- **Custom AutoAttack:** The evaluation includes APGD-CE and Square Attack, not the complete standard AutoAttack ensemble.
- **Internal evaluation:** Group-disjoint internal splits do not substitute for external clinical validation.
- **Exploratory analysis:** Test results were inspected during development, limiting claims of fully independent confirmatory evaluation.
- **Statistical uncertainty:** Group-cluster bootstrap intervals depend on the correctness of the available grouping identifiers.
- **Image shifts:** Synthetic staining and corruption experiments do not fully represent real-world laboratory or scanner variation.
- **Explainability:** Grad-CAM heatmaps provide qualitative visualization rather than validated biological explanations.
- **Clinical deployment:** No prospective testing, clinical workflow evaluation, or independent diagnostic validation was performed.

Future research would benefit from verified patient-level metadata, external whole-slide datasets, independent adversarial evaluations, and prospective validation.

---

## 📌 Conclusion

This project evaluates the adversarial robustness and reliability of deep learning models for breast histopathology patch classification.

The results show that pretrained ResNet-18 and MobileNetV3-Small achieved stronger baseline classification performance than the custom CNN within the internal dataset.

However, all three baseline models demonstrated substantial vulnerability to adversarial perturbations.

The defense experiments also revealed an important distinction between apparent resistance to FGSM and robustness against stronger iterative attacks.

While FGSM adversarial training produced high accuracy under the evaluated FGSM attack, this improvement did not persist under stronger PGD evaluation.

PGD adversarial training provided substantially stronger robustness in the observed internal experiments, although it involved a reduction in clean-image accuracy.

Additional analyses involving cross-model transferability, Grad-CAM, calibration, selective prediction, and statistical uncertainty provided a broader understanding of model behavior.

Overall, the project demonstrates that **high classification accuracy alone is not sufficient to establish reliable deep learning performance**, particularly when models are evaluated under adversarial or distribution-shift conditions.

The findings contribute to a methodological understanding of adversarial vulnerability in histopathology patch classification while emphasizing the need for careful validation, transparent reporting, and appropriately limited conclusions.

---

## 📚 References & Data Sources

1. **Breast Histopathology Images Dataset**  
   https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

2. **Janowczyk — Invasive Ductal Carcinoma Segmentation Dataset**  
   https://andrewjanowczyk.com/use-case-6-invasive-ductal-carcinoma-idc-segmentation/

3. **Goodfellow et al. — Explaining and Harnessing Adversarial Examples**  
   https://arxiv.org/abs/1412.6572

4. **Madry et al. — Towards Deep Learning Models Resistant to Adversarial Attacks**  
   https://arxiv.org/abs/1706.06083

5. **Moosavi-Dezfooli et al. — DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks**  
   https://arxiv.org/abs/1511.04599

6. **Croce & Hein — Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks**  
   https://proceedings.mlr.press/v119/croce20b.html

7. **Athalye et al. — Obfuscated Gradients Give a False Sense of Security**  
   https://proceedings.mlr.press/v80/athalye18a.html

8. **CLAIM: 2024 Update — Checklist for Artificial Intelligence in Medical Imaging**  
   https://doi.org/10.1148/ryai.240300

---

*Research scope: Retrospective internal evaluation of IDC histopathology patch classification. The project is intended for methodological research and portfolio demonstration, not clinical diagnosis or deployment.*
