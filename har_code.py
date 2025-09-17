import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STAGE 1: DATA LOADING AND EXPLORATION
# =============================================================================

def load_uci_har_data():
    """
    Load UCI Human Activity Recognition dataset
    Download from: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
    """
    # Load training data
    X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
    y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', header=None, names=['activity'])
    
    # Load test data
    X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
    y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', header=None, names=['activity'])
    
    # Load feature names
    features = pd.read_csv('UCI HAR Dataset/features.txt', delim_whitespace=True, 
                          header=None, names=['id', 'feature_name'])
    
    # Load activity labels
    activity_labels = pd.read_csv('UCI HAR Dataset/activity_labels.txt', delim_whitespace=True,
                                 header=None, names=['id', 'activity_name'])
    
    # Set feature names
    X_train.columns = features['feature_name'].values
    X_test.columns = features['feature_name'].values
    
    # Map activity numbers to names
    activity_mapping = dict(zip(activity_labels['id'], activity_labels['activity_name']))
    y_train['activity'] = y_train['activity'].map(activity_mapping)
    y_test['activity'] = y_test['activity'].map(activity_mapping)
    
    return X_train, X_test, y_train['activity'], y_test['activity'], features, activity_labels

def exploratory_data_analysis(X_train, X_test, y_train, y_test):
    """
    Perform basic EDA on the dataset
    """
    print("=== DATASET OVERVIEW ===")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(y_train.unique())}")
    print(f"Classes: {y_train.unique()}")
    
    print("\n=== CLASS DISTRIBUTION ===")
    print("Training set:")
    print(y_train.value_counts().sort_index())
    print("\nTest set:")
    print(y_test.value_counts().sort_index())
    
    # Visualize class distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    y_train.value_counts().plot(kind='bar', ax=axes[0], title='Training Set Class Distribution')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    y_test.value_counts().plot(kind='bar', ax=axes[1], title='Test Set Class Distribution')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Feature statistics
    print("\n=== FEATURE STATISTICS ===")
    print(f"Feature mean range: [{X_train.mean().min():.4f}, {X_train.mean().max():.4f}]")
    print(f"Feature std range: [{X_train.std().min():.4f}, {X_train.std().max():.4f}]")
    
    return None

# =============================================================================
# STAGE 1: LOGISTIC REGRESSION IMPLEMENTATION
# =============================================================================

def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Preprocess the data for machine learning
    """
    # Create validation split from training data
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Validation set: {X_val_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_split, y_val, scaler

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train and evaluate Logistic Regression model
    """
    print("\n=== TRAINING LOGISTIC REGRESSION ===")
    
    # Initialize and train model
    lr_model = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = lr_model.predict(X_train)
    val_pred = lr_model.predict(X_val)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Overfitting Gap: {train_acc - val_acc:.4f}")
    
    # Detailed classification report
    print("\n=== VALIDATION SET PERFORMANCE ===")
    print(classification_report(y_val, val_pred))
    
    return lr_model, train_acc, val_acc

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y_true.unique()), 
                yticklabels=sorted(y_true.unique()))
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# =============================================================================
# STAGE 2: RANDOM FOREST COMPARISON
# =============================================================================

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train and evaluate Random Forest model
    """
    print("\n=== TRAINING RANDOM FOREST ===")
    
    # Initialize and train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = rf_model.predict(X_train)
    val_pred = rf_model.predict(X_val)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Overfitting Gap: {train_acc - val_acc:.4f}")
    
    # Detailed classification report
    print("\n=== VALIDATION SET PERFORMANCE ===")
    print(classification_report(y_val, val_pred))
    
    return rf_model, train_acc, val_acc

def compare_models(lr_model, rf_model, X_val, y_val, feature_names):
    """
    Compare the two models and analyze results
    """
    print("\n=== MODEL COMPARISON ===")
    
    # Get predictions
    lr_pred = lr_model.predict(X_val)
    rf_pred = rf_model.predict(X_val)
    
    # Calculate accuracies
    lr_acc = accuracy_score(y_val, lr_pred)
    rf_acc = accuracy_score(y_val, rf_pred)
    
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"Improvement: {rf_acc - lr_acc:.4f}")
    
    # Feature importance analysis (Random Forest only)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 10 MOST IMPORTANT FEATURES (Random Forest) ===")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return lr_acc, rf_acc

def final_evaluation(best_model, X_test, y_test, model_name):
    """
    Evaluate the best model on test set
    """
    print(f"\n=== FINAL EVALUATION - {model_name} ===")
    
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\n=== TEST SET PERFORMANCE ===")
    print(classification_report(y_test, test_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, test_pred, f"Test Set Confusion Matrix - {model_name}")
    
    return test_acc

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the complete analysis
    """
    print("Starting Human Activity Recognition Analysis...")
    
    # Load data
    X_train, X_test, y_train, y_test, features, activity_labels = load_uci_har_data()
    
    # EDA
    exploratory_data_analysis(X_train, X_test, y_train, y_test)
    
    # Preprocess
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_split, y_val, scaler = preprocess_data(
        X_train, X_test, y_train, y_test
    )
    
    # Stage 1: Logistic Regression
    lr_model, lr_train_acc, lr_val_acc = train_logistic_regression(
        X_train_scaled, y_train_split, X_val_scaled, y_val
    )
    
    # Stage 2: Random Forest
    rf_model, rf_train_acc, rf_val_acc = train_random_forest(
        X_train_scaled, y_train_split, X_val_scaled, y_val
    )
    
    # Model Comparison
    lr_acc, rf_acc = compare_models(lr_model, rf_model, X_val_scaled, y_val, features['feature_name'].values)
    
    # Choose best model
    if rf_acc > lr_acc:
        best_model = rf_model
        model_name = "Random Forest"
    else:
        best_model = lr_model
        model_name = "Logistic Regression"
    
    # Final evaluation
    test_accuracy = final_evaluation(best_model, X_test_scaled, y_test, model_name)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best Model: {model_name}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()