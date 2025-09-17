#!/usr/bin/env python3
"""
Build script for Human Activity Recognition project
Handles dataset download and project execution
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

def download_uci_har_dataset():
    """
    Download the UCI HAR Dataset if it doesn't exist
    """
    dataset_path = Path("UCI HAR Dataset")
    
    if dataset_path.exists():
        print("✓ UCI HAR Dataset already exists")
        return True
    
    print("📥 Downloading UCI HAR Dataset...")
    
    # UCI HAR Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    
    try:
        # Download the dataset
        zip_path = "UCI_HAR_Dataset.zip"
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the dataset
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up zip file
        os.remove(zip_path)
        
        print("✓ UCI HAR Dataset downloaded and extracted successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please manually download the dataset from:")
        print("https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones")
        print("Extract it to the current directory as 'UCI HAR Dataset'")
        return False

def install_dependencies():
    """
    Install required dependencies
    """
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def run_analysis():
    """
    Run the main analysis script
    """
    print("🚀 Running Human Activity Recognition Analysis...")
    try:
        subprocess.check_call([sys.executable, "har_code.py"])
        print("✓ Analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analysis: {e}")
        return False

def main():
    """
    Main build function
    """
    print("🔨 Building Human Activity Recognition Project...")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Build failed: Could not install dependencies")
        return False
    
    # Step 2: Download dataset
    if not download_uci_har_dataset():
        print("❌ Build failed: Could not download dataset")
        return False
    
    # Step 3: Run analysis
    if not run_analysis():
        print("❌ Build failed: Analysis execution failed")
        return False
    
    print("=" * 50)
    print("✅ Build completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
