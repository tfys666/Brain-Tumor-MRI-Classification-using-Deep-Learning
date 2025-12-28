# Brain Tumor MRI Classification using Deep Learning


This repository contains a PyTorch implementation for automated brain tumor detection from MRI images using deep learning models. The project compares three different CNN architectures on a small medical imaging dataset and includes comprehensive evaluation metrics and visualization techniques.

## Key Features

- Complete medical image preprocessing pipeline (skull stripping, normalization)
- Data augmentation for small dataset regularization
- Implementation and comparison of three deep learning models:
  - Custom SimpleCNN (baseline)
  - ResNet18 (with transfer learning)
  - EfficientNet-B0 (with transfer learning)
- Comprehensive evaluation metrics (accuracy, recall, F1-score, AUC-ROC)
- Model interpretability using Grad-CAM visualizations
- Comparison with traditional machine learning approaches (SVM, Random Forest)

## Dataset

The project uses the Kaggle "Brain MRI Images for Brain Tumor Detection" dataset:
- 253 T1-weighted contrast-enhanced MRI images
- Binary classification task: tumor vs. no tumor
- 155 images with tumors, 98 images without tumors
- Data split: 70% training, 15% validation, 15% testing

## Performance Results

| Model | Accuracy | Recall | F1-Score | AUC |
|-------|----------|--------|----------|-----|
| SimpleCNN | 0.74 | 0.80 | 0.80 | 0.85 |
| ResNet18 | 0.87 | 0.88 | 0.90 | 0.96 |
| EfficientNet-B0 | **0.95** | **1.00** | **0.96** | 0.94 |

EfficientNet-B0 achieved the best performance with 94.9% accuracy and perfect 100% recall (zero false negatives).

## Dependencies

- Python 3.8+
- PyTorch 1.12+
- torchvision 0.13+
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- tqdm
- pandas
- pytorch-grad-cam

## Output
The script will generate:
- Training/validation curves for each model
- Confusion matrices
- ROC curves
- Grad-CAM visualizations showing model attention areas
- Performance comparison metrics
- Trained model weights

## Limitations & Future Work
- Small dataset size limits model robustness
- Single MRI slice analysis (future work: 3D volumetric analysis)
- Single MRI sequence used (future work: multi-sequence fusion)
- Single-center data (future work: multi-center validation)

## Acknowledgements
This project uses the publicly available brain tumor MRI dataset from Kaggle created by Navoneel Chakrabarty.

## License
This project is for educational and research purposes only. Not intended for clinical use.
