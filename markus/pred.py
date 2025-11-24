"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

import sys
import csv
import json

import numpy as np
import pandas as pd
from encoding import load_text_vocab, preprocess_row  # Use saved training vocab

# --- Forest prediction logic ---

def load_forest_json(json_path):
    with open(json_path, 'r') as f:
        forest = json.load(f)
    return forest

def traverse_tree(tree, features):
    node = 0
    while True:
        left = tree['children_left'][node]
        right = tree['children_right'][node]
        feature = tree['feature'][node]
        threshold = tree['threshold'][node]
        if feature == -2:  # leaf node
            value = tree['value'][node][0]
            return value
        if features[feature] <= threshold:
            node = left
        else:
            node = right

def predict_row_forest(forest, features):
    class_votes = np.zeros(forest['n_classes'], dtype=float)
    for tree in forest['trees']:
        value = traverse_tree(tree, features)
        class_votes += value
    # pick class with highest total votes
    idx = np.argmax(class_votes)
    return forest['classes'][idx]

# --- Main prediction API ---

FOREST_JSON_PATH = 'forest_model.json' 

def process_data(row):
    """Batch process entire DataFrame with encoding.preprocess (returns X,y)."""
    import encoding as en     # doesn't use sklearn, etc.
    # row_df = pd.DataFrame([row])
    X = en.preprocess(row)    
    return X

def predict_all(filename):
    """Return (predictions, labels) for the entire file using row-by-row processing."""
    forest = load_forest_json(FOREST_JSON_PATH)
    
    # Load the saved training vocabulary
    # print("Loading training vocabulary")
    vocab = load_text_vocab('text_vocab.json')
    
    df = pd.read_csv(filename)
    # Drop rows with NaN values (same as batch preprocessing)
    df = df.dropna().copy()
    # y = df['label'].values
    
    expected = forest['n_features']
    
    # Make predictions row by row
    # print(f"Processing {len(df)} rows...")
    predictions = []
    for i, row in df.iterrows():
        # Process single row using saved vocab
        features = preprocess_row(row, vocab)
        
        # Verify shape (debug)
        if len(features) != expected:
            raise ValueError(f"Feature mismatch: expected {expected}, got {len(features)}")
        
        # Make prediction
        pred = predict_row_forest(forest, features)
        predictions.append(pred)
        
        # if (i + 1) % 100 == 0:
        #     print(f"  Processed {i + 1}/{len(df)} rows...")
    
    return predictions