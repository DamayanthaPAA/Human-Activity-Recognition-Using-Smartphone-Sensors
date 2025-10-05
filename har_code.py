"""
Human Activity Recognition Using Smartphone Sensors
Complete Stage 2 Implementation - Final Version

Course: CS-C3240 Machine Learning D
Stage 2: Comparison of Logistic Regression vs Random Forest

This script implements complete end-to-end analysis:
1. Data loading and exploration
2. Preprocessing and feature scaling
3. Method 1: Logistic Regression
4. Method 2: Random Forest  
5. Model comparison and analysis
6. Final test set evaluation

Dataset: UCI Human Activity Recognition Using Smartphones
Download from: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)
import warnings
import time
import os

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
VALIDATION_SPLIT = 0.20
N_ESTIMATORS = 100

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)

# Create results directory
os.makedirs('results', exist_ok=True)

print("="*80)
print("HUMAN ACTIVITY RECOGNITION - COMPLETE ANALYSIS")
print("Stage 2: Logistic Regression vs Random Forest Comparison")
print("="*80)

#==============================================================================
# PART 1: DATA LOADING
#==============================================================================

def load_uci_har_data():
    """
    Load UCI HAR dataset from local directory.
    
    Expected directory structure:
        UCI HAR Dataset/
            train/
                X_train.txt (7352 samples × 561 features)
                y_train.txt (7352 labels)
            test/
                X_test.txt (2947 samples × 561 features)
                y_test.txt (2947 labels)
            features.txt (561 feature names)
            activity_labels.txt (6 activity mappings)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, features, activity_labels)
    """
    print("\n[1/8] Loading Dataset...")
    
    try:
        # Load training data
        X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', 
                             delim_whitespace=True, header=None)
        y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', 
                             header=None, names=['activity'])
        
        # Load test data
        X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', 
                            delim_whitespace=True, header=None)
        y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', 
                            header=None, names=['activity'])
        
        # Load feature names
        features = pd.read_csv('UCI HAR Dataset/features.txt', 
                              delim_whitespace=True,
                              header=None, names=['id', 'feature_name'])
        
        # Load activity labels
        activity_labels = pd.read_csv('UCI HAR Dataset/activity_labels.txt', 
                                     delim_whitespace=True,
                                     header=None, names=['id', 'activity_name'])
        
        # Assign feature names to columns
        X_train.columns = features['feature_name'].values
        X_test.columns = features['feature_name'].values
        
        # Map activity IDs to names
        activity_mapping = dict(zip(activity_labels['id'], 
                                   activity_labels['activity_name']))
        y_train['activity'] = y_train['activity'].map(activity_mapping)
        y_test['activity'] = y_test['activity'].map(activity_mapping)
        
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Activity classes: {len(activity_labels)}")
        
        return (X_train, X_test, y_train['activity'], y_test['activity'], 
                features, activity_labels)
    
    except FileNotFoundError:
        print("\n  ERROR: Dataset not found!")
        print("  Please download the UCI HAR Dataset and extract it to:")
        print("  ./UCI HAR Dataset/")
        print("\n  Download from:")
        print("  https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones")
        raise

#==============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
#==============================================================================

def exploratory_data_analysis(X_train, X_test, y_train, y_test):
    """
    Perform EDA to understand dataset characteristics.
    Generates visualizations and statistical summaries.
    """
    print("\n[2/8] Exploratory Data Analysis...")
    
    # Class distribution
    print("\n  Class Distribution:")
    train_dist = y_train.value_counts().sort_index()
    for activity, count in train_dist.items():
        percentage = (count / len(y_train)) * 100
        print(f"    {activity:20s}: {count:4d} ({percentage:5.2f}%)")
    
    # Visualize class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    train_dist.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_title('Training Set - Activity Distribution', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of Samples', fontsize=11)
    axes[0].set_xlabel('Activity Class', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    test_dist = y_test.value_counts().sort_index()
    test_dist.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_title('Test Set - Activity Distribution', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Number of Samples', fontsize=11)
    axes[1].set_xlabel('Activity Class', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
    print("\n  Saved: results/class_distribution.png")
    plt.close()
    
    # Feature statistics
    print(f"\n  Feature Statistics:")
    print(f"    Mean range: [{X_train.mean().min():.4f}, {X_train.mean().max():.4f}]")
    print(f"    Std range: [{X_train.std().min():.4f}, {X_train.std().max():.4f}]")
    print(f"    Missing values: {X_train.isnull().sum().sum()}")

#==============================================================================
# PART 3: DATA PREPROCESSING
#==============================================================================

def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Preprocess data:
    1. Create train/validation split from training data
    2. Apply standard scaling (z-score normalization)
    
    Returns:
        Scaled train, validation, test sets and scaler object
    """
    print(f"\n[3/8] Data Preprocessing...")
    
    # Create validation split
    print(f"\n  Creating {int((1-VALIDATION_SPLIT)*100)}/{int(VALIDATION_SPLIT*100)} train/validation split...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_STATE, 
        stratify=y_train
    )
    
    print(f"    Training set: {X_train_split.shape[0]} samples")
    print(f"    Validation set: {X_val.shape[0]} samples")
    print(f"    Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    print(f"\n  Applying standard scaling (z-score normalization)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"    Features scaled to mean=0, std=1")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_split, y_val, scaler

#==============================================================================
# PART 4: LOGISTIC REGRESSION
#==============================================================================

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train and evaluate Logistic Regression classifier.
    
    Method: One-vs-Rest multi-class logistic regression
    Loss: Logistic loss (cross-entropy) with L2 regularization
    """
    print("\n[4/8] Training Logistic Regression...")
    print("  Strategy: One-vs-Rest (OvR)")
    print("  Loss function: Logistic loss (cross-entropy)")
    print("  Regularization: L2 (C=1.0)")
    
    # Train model
    start_time = time.time()
    lr_model = LogisticRegression(
        multi_class='ovr', 
        max_iter=1000, 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    lr_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    train_pred = lr_model.predict(X_train)
    val_pred = lr_model.predict(X_val)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"\n  Results:")
    print(f"    Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"    Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"    Overfitting Gap: {train_acc - val_acc:.4f} ({(train_acc - val_acc)*100:.2f}%)")
    print(f"    Training Time: {training_time:.2f}s")
    print(f"    Prediction Time: {prediction_time:.3f}s")
    
    # Detailed performance
    print("\n  Per-Class Performance (Validation):")
    report = classification_report(y_val, val_pred, output_dict=True)
    for activity in sorted(y_val.unique()):
        if activity in report:
            print(f"    {activity:20s}: Precision={report[activity]['precision']:.3f}, "
                  f"Recall={report[activity]['recall']:.3f}, F1={report[activity]['f1-score']:.3f}")
    
    return lr_model, train_acc, val_acc, training_time, prediction_time

#==============================================================================
# PART 5: RANDOM FOREST
#==============================================================================

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train and evaluate Random Forest classifier.
    
    Method: Ensemble of decision trees
    Criterion: Gini impurity for split quality
    """
    print(f"\n[5/8] Training Random Forest...")
    print(f"  Number of trees: {N_ESTIMATORS}")
    print(f"  Split criterion: Gini impurity")
    print(f"  Bootstrap sampling: Yes")
    
    # Train model
    start_time = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    train_pred = rf_model.predict(X_train)
    val_pred = rf_model.predict(X_val)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"\n  Results:")
    print(f"    Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"    Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"    Overfitting Gap: {train_acc - val_acc:.4f} ({(train_acc - val_acc)*100:.2f}%)")
    print(f"    Training Time: {training_time:.2f}s")
    print(f"    Prediction Time: {prediction_time:.3f}s")
    
    # Detailed performance
    print("\n  Per-Class Performance (Validation):")
    report = classification_report(y_val, val_pred, output_dict=True)
    for activity in sorted(y_val.unique()):
        if activity in report:
            print(f"    {activity:20s}: Precision={report[activity]['precision']:.3f}, "
                  f"Recall={report[activity]['recall']:.3f}, F1={report[activity]['f1-score']:.3f}")
    
    return rf_model, train_acc, val_acc, training_time, prediction_time

#==============================================================================
# PART 6: MODEL COMPARISON
#==============================================================================

def compare_models(lr_model, rf_model, X_val, y_val, feature_names,
                  lr_times, rf_times):
    """
    Compare Logistic Regression vs Random Forest.
    Analyzes accuracy, per-class performance, and computational efficiency.
    """
    print("\n[6/8] Model Comparison...")
    
    # Get predictions
    lr_pred = lr_model.predict(X_val)
    rf_pred = rf_model.predict(X_val)
    
    # Calculate accuracies
    lr_acc = accuracy_score(y_val, lr_pred)
    rf_acc = accuracy_score(y_val, rf_pred)
    
    print(f"\n  Validation Accuracy Comparison:")
    print(f"    Logistic Regression: {lr_acc:.4f} ({lr_acc*100:.2f}%)")
    print(f"    Random Forest:       {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print(f"    Difference:          {lr_acc - rf_acc:+.4f} ({(lr_acc - rf_acc)*100:+.2f}%)")
    
    # Per-class F1 comparison
    print(f"\n  Per-Class F1-Score Comparison:")
    lr_report = classification_report(y_val, lr_pred, output_dict=True)
    rf_report = classification_report(y_val, rf_pred, output_dict=True)
    
    print(f"\n    {'Activity':<25} {'LR F1':>8} {'RF F1':>8} {'Diff':>8}")
    print(f"    {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for activity in sorted(y_val.unique()):
        if activity in lr_report and activity in rf_report:
            lr_f1 = lr_report[activity]['f1-score']
            rf_f1 = rf_report[activity]['f1-score']
            diff = lr_f1 - rf_f1
            print(f"    {activity:<25} {lr_f1:>8.4f} {rf_f1:>8.4f} {diff:>+8.4f}")
    
    # Computational efficiency
    print(f"\n  Computational Efficiency:")
    print(f"    {'Metric':<30} {'LR':>12} {'RF':>12} {'Ratio':>12}")
    print(f"    {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"    {'Training Time (s)':<30} {lr_times[0]:>12.2f} {rf_times[0]:>12.2f} {rf_times[0]/lr_times[0]:>11.1f}x")
    print(f"    {'Prediction Time (s)':<30} {lr_times[1]:>12.3f} {rf_times[1]:>12.3f} {rf_times[1]/lr_times[1]:>11.1f}x")
    
    # Feature importance (Random Forest)
    print(f"\n  Feature Importance Analysis (Random Forest):")
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n    Top 10 Most Important Features:")
    for idx, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
        print(f"      {idx:2d}. {row['feature']:<45} {row['importance']:.6f}")
    
    # Save feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    plt.barh(range(len(top_features)), top_features['importance'], color=colors, edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n  Saved: results/feature_importance.png")
    plt.close()
    
    # Confusion matrices comparison
    plot_confusion_matrices(y_val, lr_pred, rf_pred)
    
    return lr_acc, rf_acc, feature_importance_df

def plot_confusion_matrices(y_val, lr_pred, rf_pred):
    """Plot confusion matrices for both models side by side."""
    activities = sorted(y_val.unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Logistic Regression
    lr_cm = confusion_matrix(y_val, lr_pred, labels=activities)
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activities, yticklabels=activities, ax=axes[0],
                cbar_kws={'label': 'Count'}, vmin=0)
    axes[0].set_title('Logistic Regression - Confusion Matrix', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Random Forest
    rf_cm = confusion_matrix(y_val, rf_pred, labels=activities)
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=activities, yticklabels=activities, ax=axes[1],
                cbar_kws={'label': 'Count'}, vmin=0)
    axes[1].set_title('Random Forest - Confusion Matrix', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: results/confusion_matrices_comparison.png")
    plt.close()

#==============================================================================
# PART 7: FINAL TEST EVALUATION
#==============================================================================

def final_test_evaluation(best_model, X_test, y_test, model_name):
    """
    Evaluate selected model on held-out test set.
    Provides unbiased performance estimate.
    """
    print(f"\n[7/8] Final Test Evaluation - {model_name}...")
    
    # Predict on test set
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Detailed per-class performance
    print("\n  Per-Class Performance (Test Set):")
    report = classification_report(y_test, test_pred, output_dict=True)
    for activity in sorted(y_test.unique()):
        if activity in report:
            print(f"    {activity:20s}: Precision={report[activity]['precision']:.4f}, "
                  f"Recall={report[activity]['recall']:.4f}, F1={report[activity]['f1-score']:.4f}")
    
    # Confusion matrix
    activities = sorted(y_test.unique())
    cm = confusion_matrix(y_test, test_pred, labels=activities)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                xticklabels=activities, yticklabels=activities,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Test Set Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n  Saved: results/test_confusion_matrix.png")
    plt.close()
    
    # Save full classification report
    with open('results/classification_report.txt', 'w') as f:
        f.write(f"Test Set Classification Report - {model_name}\n")
        f.write("="*60 + "\n\n")
        f.write(classification_report(y_test, test_pred))
    print("  Saved: results/classification_report.txt")
    
    return test_acc

#==============================================================================
# PART 8: GENERATE SUMMARY
#==============================================================================

def generate_summary(results):
    """Generate and save analysis summary."""
    print("\n[8/8] Generating Summary Report...")
    
    summary = f"""
{'='*80}
HUMAN ACTIVITY RECOGNITION - ANALYSIS SUMMARY
{'='*80}

DATASET INFORMATION:
  - Total samples: {results['total_samples']}
  - Training samples: {results['train_samples']}
  - Validation samples: {results['val_samples']}
  - Test samples: {results['test_samples']}
  - Features: {results['n_features']}
  - Classes: {results['n_classes']}

METHOD 1: LOGISTIC REGRESSION
  - Training Accuracy: {results['lr_train_acc']:.4f} ({results['lr_train_acc']*100:.2f}%)
  - Validation Accuracy: {results['lr_val_acc']:.4f} ({results['lr_val_acc']*100:.2f}%)
  - Overfitting Gap: {results['lr_train_acc'] - results['lr_val_acc']:.4f}
  - Training Time: {results['lr_train_time']:.2f}s
  - Prediction Time: {results['lr_pred_time']:.3f}s

METHOD 2: RANDOM FOREST
  - Training Accuracy: {results['rf_train_acc']:.4f} ({results['rf_train_acc']*100:.2f}%)
  - Validation Accuracy: {results['rf_val_acc']:.4f} ({results['rf_val_acc']*100:.2f}%)
  - Overfitting Gap: {results['rf_train_acc'] - results['rf_val_acc']:.4f}
  - Training Time: {results['rf_train_time']:.2f}s
  - Prediction Time: {results['rf_pred_time']:.3f}s

MODEL COMPARISON:
  - Best Method: {results['best_model_name']}
  - Validation Accuracy Difference: {results['lr_val_acc'] - results['rf_val_acc']:+.4f}
  - Speed Advantage (Training): {results['rf_train_time']/results['lr_train_time']:.1f}x faster (LR)
  - Speed Advantage (Prediction): {results['rf_pred_time']/results['lr_pred_time']:.1f}x faster (LR)

FINAL TEST RESULTS:
  - Selected Model: {results['best_model_name']}
  - Test Accuracy: {results['test_acc']:.4f} ({results['test_acc']*100:.2f}%)
  - Generalization Gap: {abs(results['best_val_acc'] - results['test_acc']):.4f}

CONCLUSION:
  {results['conclusion']}

GENERATED FILES:
  - results/class_distribution.png
  - results/feature_importance.png
  - results/confusion_matrices_comparison.png
  - results/test_confusion_matrix.png
  - results/classification_report.txt
  - results/analysis_summary.txt

{'='*80}
Analysis completed successfully!
{'='*80}
"""
    
    print(summary)
    
    # Save to file
    with open('results/analysis_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\n  Saved: results/analysis_summary.txt")

#==============================================================================
# MAIN EXECUTION
#==============================================================================

def main():
    """Main execution pipeline."""
    
    # Load data
    X_train, X_test, y_train, y_test, features, activity_labels = load_uci_har_data()
    
    # EDA
    exploratory_data_analysis(X_train, X_test, y_train, y_test)
    
    # Preprocess
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_split, y_val, scaler = preprocess_data(
        X_train, X_test, y_train, y_test
    )
    
    # Train Logistic Regression
    lr_model, lr_train_acc, lr_val_acc, lr_train_time, lr_pred_time = train_logistic_regression(
        X_train_scaled, y_train_split, X_val_scaled, y_val
    )
    
    # Train Random Forest
    rf_model, rf_train_acc, rf_val_acc, rf_train_time, rf_pred_time = train_random_forest(
        X_train_scaled, y_train_split, X_val_scaled, y_val
    )
    
    # Compare models
    lr_acc, rf_acc, feature_importance = compare_models(
        lr_model, rf_model, X_val_scaled, y_val,
        features['feature_name'].values,
        (lr_train_time, lr_pred_time),
        (rf_train_time, rf_pred_time)
    )
    
    # Select best model
    if lr_acc > rf_acc:
        best_model = lr_model
        best_model_name = "Logistic Regression"
        best_val_acc = lr_acc
    else:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_val_acc = rf_acc
    
    print(f"\n  Selected Best Model: {best_model_name}")
    
    # Final test evaluation
    test_acc = final_test_evaluation(best_model, X_test_scaled, y_test, best_model_name)
    
    # Generate summary
    results = {
        'total_samples': len(X_train) + len(X_test),
        'train_samples': len(X_train_scaled),
        'val_samples': len(X_val_scaled),
        'test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'n_classes': len(activity_labels),
        'lr_train_acc': lr_train_acc,
        'lr_val_acc': lr_val_acc,
        'lr_train_time': lr_train_time,
        'lr_pred_time': lr_pred_time,
        'rf_train_acc': rf_train_acc,
        'rf_val_acc': rf_val_acc,
        'rf_train_time': rf_train_time,
        'rf_pred_time': rf_pred_time,
        'best_model_name': best_model_name,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'conclusion': f"{best_model_name} achieved {test_acc*100:.2f}% test accuracy, "
                     f"demonstrating {'strong' if test_acc > 0.95 else 'good'} generalization for activity recognition."
    }
    
    generate_summary(results)
    
    return results

if __name__ == "__main__":
    results = main()