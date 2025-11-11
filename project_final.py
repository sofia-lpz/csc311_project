"""
This Python file provides some useful code for reading the training file
"clean_dataset.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

import numpy as np
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer


file_name = "training_data_clean.csv"


def process_multiselect(series, target_tasks):
    """Convert multiselect strings to lists, keeping only specified features"""
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            # Check which of the target tasks are present in the response
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    """
    Extract numeric rating from responses like '3 - Sometimes'.
    Returns None for missing responses
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None

def preprocess(df):
    # Drop rows with missing data
    df.dropna(inplace=True)

    # Define the tasks we want to use as features (first four, clean ones)
    target_tasks = [
        'Math computations',
        'Writing or debugging code',
        'Data processing or analysis', 
        'Explaining complex concepts simply',
    ]

    # Process multi-select columns (use exact column names from your cleaned data)
    best_tasks_lists = process_multiselect(df['Which types of tasks do you feel this model handles best? (Select all that apply.)'], target_tasks)
    suboptimal_tasks_lists = process_multiselect(df['For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'], target_tasks)
    
    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()
    
    best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
    suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)

    # Use some rating features
    academic_numeric = df['How likely are you to use this model for academic tasks?'].apply(extract_rating)
    subopt_numeric = df['Based on your experience, how often has this model given you a response that felt suboptimal?'].apply(extract_rating)

    # Combine features
    X = np.hstack([academic_numeric.values.reshape(-1, 1), subopt_numeric.values.reshape(-1, 1), 
                   best_tasks_encoded, suboptimal_tasks_encoded])
    y = df['label'].values

    return X, y

def one_hot_encode(df, max_features=50, use_tfidf=False):
    text_columns = [
        "In your own words, what kinds of tasks would you use this model for?",
        "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
        "When you verify a response from this model, how do you usually go about it?"
    ]
    
    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
    
    encoded_features = []
    
    for col in text_columns:
        if col in df.columns:
            text_data = df[col].fillna('')
            
            vectorizer = Vectorizer(
                max_features=max_features,
                stop_words='english',  # Remove common words like 'the', 'is', etc.
                min_df=2,  # Word must appear in at least 2 documents (exludes typos and whatnot)
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3+ letters
            )
            
            features = vectorizer.fit_transform(text_data)
            encoded_features.append(features.toarray())
    
    if encoded_features:
        return np.hstack(encoded_features)
    else:
        return np.array([]).reshape(len(df), 0)
    
def split(df, train_ratio=0.8, val_ratio=0.10, test_ratio=0.10, random_state=42, output_dir="./datasets"):
    unique_students = df['student_id'].unique()
    n_students = len(unique_students)
    
    np.random.seed(random_state)
    shuffled_students = np.random.permutation(unique_students)
    
    n_train = int(n_students * train_ratio)
    n_val = int(n_students * val_ratio)
    
    train_students = shuffled_students[:n_train]
    val_students = shuffled_students[n_train:n_train + n_val]
    test_students = shuffled_students[n_train + n_val:]
    
    train_df = df[df['student_id'].isin(train_students)].copy()
    val_df = df[df['student_id'].isin(val_students)].copy()
    test_df = df[df['student_id'].isin(test_students)].copy()
    
    assert len(set(train_students) & set(val_students)) == 0, "Student leak between train and val!"
    assert len(set(train_students) & set(test_students)) == 0, "Student leak between train and test!"
    assert len(set(val_students) & set(test_students)) == 0, "Student leak between val and test!"
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return train_df, val_df, test_df

def main():
    # Load processed data
    df = pd.read_csv(file_name)

    split(df, output_dir="./datasets")

    X, y = preprocess(df)

    # Simple train / test split
    n_train = int(0.7 * len(X))
    X_train, y_train, X_test, y_test = X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    # Train simple KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Evaluate
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

if __name__ == "__main__":
    main()