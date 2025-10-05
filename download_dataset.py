"""
Script to download and extract the UCI HAR Dataset
"""

import os
import urllib.request
import zipfile
import shutil

def download_uci_har_dataset():
    """
    Download and extract the UCI HAR Dataset
    """
    print("Downloading UCI HAR Dataset...")
    
    # Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    
    # Download the dataset
    print("Downloading dataset from UCI ML Repository...")
    urllib.request.urlretrieve(url, "UCI_HAR_Dataset.zip")
    print("Download completed!")
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile("UCI_HAR_Dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # Clean up the zip file
    os.remove("UCI_HAR_Dataset.zip")
    print("Dataset extracted successfully!")
    print("You can now run: python har_code.py")

if __name__ == "__main__":
    download_uci_har_dataset()
