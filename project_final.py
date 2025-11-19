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

all_multiselect_tasks = [
    'Brainstorming or generating creative ideas',
    'Converting content between formats (e.g. LaTeX)',
    'Drafting professional text (e.g. emails, résumés)',
    'Math computations',
    'Writing or debugging code',
    'Data processing or analysis', 
    'Explaining complex concepts simply',
    'Writing or editing essays/reports',
]

def preprocess(df, max_features=50, 
               use_tfidf=False, 
               feature_combo = ('ratings', 'best_tasks', 'subopt_tasks', 'text'), 
               multiselect_tasks=all_multiselect_tasks):
    
    # Drop rows with missing data
    df.dropna(inplace=True)

    # Build feature matrix
    X = build_features(
        df, 
        feature_combo, 
        multiselect_tasks=multiselect_tasks, 
        max_features=max_features, 
        use_tfidf=use_tfidf
    )

    y = df['label'].values

    return X, y
  
def hyperparameter_tuning_knn(X_train, y_train, X_val, y_val, k_values):
    best_k = None
    best_val_acc = 0.0
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        val_acc = knn.score(X_val, y_val)
        
        print(f"Validation accuracy for k={k}: {val_acc:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k
            
    print(f"Best k: {best_k} with validation accuracy: {best_val_acc:.3f}")
    return best_k

def build_features(df, feature_combo, multiselect_tasks=all_multiselect_tasks, max_features=50, use_tfidf=False):
    """
    Build feature matrix based on specified feature combination.
    
    Args:
        df: DataFrame with data
        feature_combo: tuple of feature group names to include
        max_features: max features for text vectorization
        use_tfidf: whether to use TF-IDF
        multiselect_tasks: list of tasks for multi-select processing
    Returns:
        Feature matrix (numpy array)
    """
    features_list = []
    
    # Rating features
    if 'ratings' in feature_combo:
        academic_numeric = df['How likely are you to use this model for academic tasks?'].apply(extract_rating)
        subopt_numeric = df['Based on your experience, how often has this model given you a response that felt suboptimal?'].apply(extract_rating)
        references_numeric = df['How often do you expect this model to provide responses with references or supporting evidence?'].apply(extract_rating)
        verified_numeric = df['How often do you verify this model\'s responses?'].apply(extract_rating)
        
        features_list.extend([
            academic_numeric.values.reshape(-1, 1),
            subopt_numeric.values.reshape(-1, 1),
            references_numeric.values.reshape(-1, 1),
            verified_numeric.values.reshape(-1, 1)
        ])
    
    # Multi-select features
    if 'best_tasks' in feature_combo:
        best_tasks_lists = process_multiselect(
            df['Which types of tasks do you feel this model handles best? (Select all that apply.)'], 
            multiselect_tasks
        )
        mlb_best = MultiLabelBinarizer()
        best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
        features_list.append(best_tasks_encoded)
    
    if 'subopt_tasks' in feature_combo:
        suboptimal_tasks_lists = process_multiselect(
            df['For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'], 
            multiselect_tasks
        )
        mlb_subopt = MultiLabelBinarizer()
        suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)
        features_list.append(suboptimal_tasks_encoded)
    
    # Text features
    if 'text' in feature_combo:
        text_features = one_hot_encode(df, max_features=max_features, use_tfidf=use_tfidf)
        features_list.append(text_features)
    
    # Combine all selected features
    if features_list:
        return np.hstack(features_list)
    else:
        return np.array([]).reshape(len(df), 0)

def feature_combination_testing_knn():
    # tests all combinations of feature groups and possible multiselect_tasks for kNN performance
    pass

"""Helper functions"""

def one_hot_encode(df, max_features=50, use_tfidf=False):

    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer

    text_columns = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
    ]
    
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
    
def process_multiselect(series, multiselect_tasks):
    """Convert multiselect strings to lists, keeping only specified features"""
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            # Check which of the target tasks are present in the response
            present_tasks = [task for task in multiselect_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed

def extract_rating(response):
    """
    Extract numeric rating from responses like '3 - Sometimes'.
    Returns None for missing responses
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None

def split(df, train_ratio=0.8, val_ratio=0.05, random_state=42, output_dir="./datasets"):
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
    
    return train_df, val_df, test_df

def main():
    # Load processed data
    df = pd.read_csv(file_name)
    
    # Split into train/val/test datasets
    train_df, val_df, test_df = split(df, output_dir="./datasets")
    
    # Preprocess each split to get features (X) and labels (y)
    X_train, y_train = preprocess(train_df, max_features=50, use_tfidf=False)
    X_val, y_val = preprocess(val_df, max_features=50, use_tfidf=False)
    X_test, y_test = preprocess(test_df, max_features=50, use_tfidf=False)
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Hyperparameter tuning for k in kNN
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    best_k = hyperparameter_tuning_knn(X_train, y_train, X_val, y_val, k_values)

    # Train the best kNN model
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    
    # Evaluate on all splits
    train_acc = knn.score(X_train, y_train)
    val_acc = knn.score(X_val, y_val)
    test_acc = knn.score(X_test, y_test)
    
    print(f"\nTraining accuracy: {train_acc:.3f}")
    print(f"Validation accuracy: {val_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

if __name__ == "__main__":
    main()