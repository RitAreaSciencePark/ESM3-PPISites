import os, sys, random
import json, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Load dataset
# -------------------------------
"""
def load_biodl_dataset(csv_file, dataset_type):
    label_column_name = f"{dataset_type}_interface"
    df = pd.read_csv(csv_file)[["sequence", label_column_name]]
    df = df.applymap(lambda x: str(x).replace(",", ""))

    seqs = df.sequence.to_list()
    labels_str = df[label_column_name].to_list()
    labels = [list(map(int, list(s))) for s in labels_str]
    return seqs, labels
"""

def load_biodl_dataset(csv_file, dataset_type):
    label_column_name = f"{dataset_type}_interface"
    
    # Read CSV
    df = pd.read_csv(csv_file)[["sequence", label_column_name]]
    
    # Clean data: 
    # 1. Convert to string
    # 2. Remove commas (if any)
    # 3. Strip whitespace/newlines from the start and end of strings (CRITICAL FIX)
    df = df.applymap(lambda x: str(x).replace(",", "").strip())

    seqs = df.sequence.to_list()
    labels_str = df[label_column_name].to_list()
    
    # Map characters to integers
    # Added .strip() here as a second safety measure
    labels = [list(map(int, list(s.strip()))) for s in labels_str]
    
    return seqs, labels