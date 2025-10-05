# Human Activity Recognition Using Smartphone Sensors

A complete machine learning project for human activity recognition using smartphone sensor data. This project implements and compares Logistic Regression and Random Forest classifiers for activity classification.

## Project Overview

This project implements a comprehensive analysis pipeline for human activity recognition using the UCI HAR Dataset. The analysis includes:

1. **Data Loading and Exploration** - Load and analyze the UCI HAR dataset
2. **Data Preprocessing** - Feature scaling and train/validation split
3. **Model Training** - Logistic Regression and Random Forest classifiers
4. **Model Comparison** - Performance analysis and feature importance
5. **Final Evaluation** - Test set evaluation with comprehensive reporting

## Dataset

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**:
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
- **Activities**: 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
- **Features**: 561 time and frequency domain variables
- **Samples**: 10,299 total samples (7,352 training, 2,947 test)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Human-Activity-Recognition-Using-Smartphone-Sensors
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   ```bash
   python download_dataset.py
   ```
   
   Or manually download from: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

## Usage

### Quick Start

1. **Download the dataset** (if not already done):
   ```bash
   python download_dataset.py
   ```

2. **Run the complete analysis**:
   ```bash
   python har_code.py
   ```

### Expected Output

The script will generate the following outputs in the `results/` directory:

- `class_distribution.png` - Activity class distribution visualization
- `feature_importance.png` - Top 15 most important features (Random Forest)
- `confusion_matrices_comparison.png` - Side-by-side confusion matrices
- `test_confusion_matrix.png` - Final test set confusion matrix
- `classification_report.txt` - Detailed classification report
- `analysis_summary.txt` - Complete analysis summary

### Console Output

The script provides detailed progress information including:
- Dataset loading and statistics
- Model training progress and performance metrics
- Model comparison results
- Final test evaluation
- Comprehensive summary report

## Project Structure

```
Human-Activity-Recognition-Using-Smartphone-Sensors/
├── har_code.py              # Main analysis script
├── download_dataset.py      # Dataset download helper
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── results/                # Output directory (created automatically)
└── UCI HAR Dataset/        # Dataset directory (after download)
    ├── train/
    ├── test/
    ├── features.txt
    └── activity_labels.txt
```

## Features

### Data Analysis
- **Exploratory Data Analysis** with class distribution visualization
- **Feature statistics** and data quality assessment
- **Train/validation/test split** with proper stratification

### Model Implementation
- **Logistic Regression**: One-vs-Rest multi-class classification
- **Random Forest**: Ensemble of 100 decision trees
- **Feature scaling** using StandardScaler (z-score normalization)

### Evaluation Metrics
- **Accuracy scores** for training, validation, and test sets
- **Per-class performance** (precision, recall, F1-score)
- **Confusion matrices** for detailed error analysis
- **Feature importance** analysis (Random Forest)
- **Computational efficiency** comparison

### Visualization
- **Class distribution** plots for training and test sets
- **Feature importance** bar charts
- **Confusion matrices** with proper labeling
- **High-quality plots** saved as PNG files (300 DPI)

## Technical Details

### Dependencies
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0

### Configuration
- **Random State**: 42 (for reproducibility)
- **Validation Split**: 20% of training data
- **Random Forest Trees**: 100 estimators
- **Cross-validation**: Stratified split

### Performance Expectations
- **Training Time**: ~1-5 seconds (depending on hardware)
- **Prediction Time**: <0.1 seconds
- **Expected Accuracy**: >95% on test set
- **Memory Usage**: ~100-200 MB

## Results Interpretation

### Model Comparison
The script automatically compares both models and selects the best performer based on validation accuracy. Typical results:

- **Logistic Regression**: Fast training, good baseline performance
- **Random Forest**: Higher accuracy, better feature utilization
- **Best Model**: Automatically selected based on validation performance

### Output Files
All results are saved in the `results/` directory with descriptive filenames and high-quality formatting suitable for reports and presentations.

## Troubleshooting

### Common Issues

1. **Dataset not found**: Run `python download_dataset.py` first
2. **Memory issues**: Ensure sufficient RAM (4GB+ recommended)
3. **Unicode errors**: Fixed in the current version
4. **Missing dependencies**: Run `pip install -r requirements.txt`

### Performance Tips

- Use SSD storage for faster data loading
- Ensure sufficient RAM for large datasets
- Close other applications during training for optimal performance

## License

This project is for educational purposes. The UCI HAR Dataset is publicly available under the UCI ML Repository license.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.