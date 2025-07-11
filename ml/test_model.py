#!/usr/bin/env python3
"""
Test script for the Raga Classifier ML model using max-counts frequency vector features.
Tests both a dense neural network and classical ML models. Uses majority voting for inference.
"""

import os
import argparse
import numpy as np
from glob import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from ml.raga_classifier import RagaClassifier, extract_features_from_file

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Test Raga Classifier with max-counts features.")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to test dataset root. Subfolders = ragas.')
    parser.add_argument('--model-type', type=str, default='dense', choices=['dense', 'logreg', 'svm', 'xgb'], help='Model type to test.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model.')
    parser.add_argument('--label-encoder', type=str, required=True, help='Path to label encoder pickle.')
    return parser.parse_args()
        
# --- Data Loading ---
def load_data(data_dir):
    X, y, files = [], [], []
    ragas = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for raga in ragas:
        fpaths = glob(os.path.join(data_dir, raga, '*.wav'))
        for f in fpaths:
            feats = extract_features_from_file(f)
            X.append(feats)
            y.append(raga)
            files.append(f)
    return X, y, files, ragas

# --- Main Testing ---
def main():
    args = parse_args()
    print(f"Loading test data from {args.data_dir} ...")
    X, y, files, ragas = load_data(args.data_dir)
    print(f"Loaded {len(X)} files from {len(ragas)} ragas.")
        
    # Load label encoder
    with open(args.label_encoder, 'rb') as f:
        le = pickle.load(f)
    y_enc = le.transform(y)

    # Load model
    clf = RagaClassifier(model_type=args.model_type, model_path=args.model_path, num_classes=len(le.classes_))

    # Predict with majority voting
    y_pred = []
    for feats in X:
        pred = clf.predict(feats)
        y_pred.append(pred)
    y_pred = np.array(y_pred)

    # Report
    acc = accuracy_score(y_enc, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_enc, y_pred))
    print("Classification report:")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

if __name__ == '__main__':
    main() 