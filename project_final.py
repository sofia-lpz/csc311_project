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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import MultiLabelBinarizer

from itertools import combinations

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

"""feature testing function"""
def find_best_model(train_df, val_df, model_family='knn', max_features=50, use_tfidf=False):
    """
    Find the best model configuration by testing all feature combinations.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        model_family: 'knn', 'random_forest', or 'naive_bayes'
        max_features: Maximum features for text vectorization
        use_tfidf: Whether to use TF-IDF for text features
    
    Returns:
        dict: {
            'model': best trained model,
            'feature_combo': best feature combination,
            'accuracy': best validation accuracy,
            'hyperparameters': best hyperparameters found
        }
    """
    
    best_overall_accuracy = 0
    best_overall_model = None
    best_overall_combo = None
    best_overall_params = None

    all_feature_combos = ['ratings, best_tasks, subopt_tasks, text']
    
    print(f"\n{'='*80}")
    print(f"Testing {model_family.upper()} with {len(all_feature_combos)} feature combinations")
    print(f"{'='*80}\n")
    
    # Test each feature combination
    for i, feature_combo in enumerate(all_feature_combos, 1):
        print(f"\n--- Feature Combination {i}/{len(all_feature_combos)}: {feature_combo} ---")
        
        try:
            # Preprocess training data and get fitted encoders
            X_train, y_train, encoders = preprocess(
                train_df, 
                max_features=max_features,
                use_tfidf=use_tfidf,
                fitted_encoders=None  # Fit new encoders on training data
            )
            
            # Preprocess validation data using the same encoders
            X_val, y_val = preprocess(
                val_df, 
                max_features=max_features,
                use_tfidf=use_tfidf,
                fitted_encoders=encoders  # Use encoders from training
            )
            
            # Check if we have valid data
            if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                print(f"  Skipping: No valid samples after preprocessing")
                continue
            
            if X_train.shape[1] == 0:
                print(f"  Skipping: No features generated")
                continue
            
            print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"  Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
            
            # Train the appropriate model
            if model_family.lower() == 'knn':
                model = best_knn_model(X_train, y_train, X_val, y_val)
            elif model_family.lower() == 'random_forest':
                model = best_random_forest_model(X_train, y_train, X_val, y_val)
            elif model_family.lower() == 'naive_bayes':
                model = best_multinomial_naive_bayes_model(X_train, y_train, X_val, y_val)
            else:
                raise ValueError(f"Unknown model family: {model_family}")
            
            if model is None:
                print(f"  No valid model found for this feature combination")
                continue
            
            # Evaluate on validation set
            val_accuracy = model.score(X_val, y_val)
            
            # Update best overall model if this is better
            if val_accuracy > best_overall_accuracy:
                best_overall_accuracy = val_accuracy
                best_overall_model = model
                best_overall_combo = feature_combo
                best_overall_params = {
                    'max_features': max_features,
                    'use_tfidf': use_tfidf,
                    'n_train_samples': X_train.shape[0],
                    'n_features': X_train.shape[1]
                }
                print(f"  *** NEW BEST MODEL! Accuracy: {val_accuracy:.4f} ***")
            else:
                print(f"  Accuracy: {val_accuracy:.4f}")
                
        except Exception as e:
            print(f"  Error with feature combination {feature_combo}: {e}")
            continue
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"BEST {model_family.upper()} MODEL RESULTS")
    print(f"{'='*80}")
    if best_overall_model is not None:
        print(f"Best feature combination: {best_overall_combo}")
        print(f"Best validation accuracy: {best_overall_accuracy:.4f}")
        print(f"Number of features: {best_overall_params['n_features']}")
        print(f"Number of training samples: {best_overall_params['n_train_samples']}")
        print(f"Max text features: {best_overall_params['max_features']}")
        print(f"Using TF-IDF: {best_overall_params['use_tfidf']}")
    else:
        print("No valid model found!")
    print(f"{'='*80}\n")
    
    return {
        'model': best_overall_model,
        'feature_combo': best_overall_combo,
        'accuracy': best_overall_accuracy,
        'hyperparameters': best_overall_params
    }

"""model testing functions"""
def best_knn_model(X_train, y_train, X_val, y_val):
    """
    Find the best k for kNN based on validation accuracy.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_val: Validation feature matrix
        y_val: Validation labels

    Returns:
        Trained kNN model with best k
    """
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]  # Fixed typo from k_vales
    distances = ['euclidean', 'manhattan', 'cosine']
    weights = ['uniform', 'distance']
    
    best_accuracy = 0
    best_model = None
    best_params = {}
    
    # Grid search over all hyperparameter combinations
    for k in k_values:
        for distance in distances:
            for weight in weights:
                try:
                    # Train kNN model with current hyperparameters
                    knn = KNeighborsClassifier(
                        n_neighbors=k,
                        metric=distance,
                        weights=weight
                    )
                    knn.fit(X_train, y_train)
                    
                    # Evaluate on validation set
                    val_accuracy = knn.score(X_val, y_val)
                    
                    # Update best model if this is better
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_model = knn
                        best_params = {
                            'k': k,
                            'distance': distance,
                            'weight': weight,
                            'accuracy': val_accuracy
                        }
                        
                except Exception as e:
                    # Handle cases where certain combinations might fail
                    # (e.g., cosine distance with certain data)
                    print(f"Failed for k={k}, distance={distance}, weight={weight}: {e}")
                    continue
    
    # Print best configuration found
    if best_model is not None:
        print(f"Best kNN model found:")
        print(f"  k={best_params['k']}")
        print(f"  distance={best_params['distance']}")
        print(f"  weight={best_params['weight']}")
        print(f"  validation accuracy={best_params['accuracy']:.4f}")
    else:
        print("No valid kNN model found!")
    
    return best_model

def best_random_forest_model(X_train, y_train, X_val, y_val):
    """
    Find the best Random Forest model based on validation accuracy.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_val: Validation feature matrix
        y_val: Validation labels
    Returns:
        Trained Random Forest model with best hyperparameters
    """
    n_estimators_list = [100,200,300,500]
    max_depths = [None, 10, 20, 30]
    min_samples_splits = [2, 5, 10]
    min_samples_leafs = [1, 2, 4]
    max_features_list = ['sqrt', 'log2', None]  # 'auto' is deprecated
    criterions = ['gini', 'entropy']
    
    best_accuracy = 0
    best_model = None
    best_params = {}
    
    # Total combinations to test
    total_combos = (len(n_estimators_list) * len(max_depths) * 
                   len(min_samples_splits) * len(min_samples_leafs) * 
                   len(max_features_list) * len(criterions))
    
    print(f"Testing {total_combos} hyperparameter combinations for Random Forest...")
    
    combo_count = 0
    
    # Grid search over all hyperparameter combinations
    for n_est in n_estimators_list:
        for max_depth in max_depths:
            for min_split in min_samples_splits:
                for min_leaf in min_samples_leafs:
                    for max_feat in max_features_list:
                        for criterion in criterions:
                            combo_count += 1
                            
                            # Print progress every 50 combinations
                            if combo_count % 50 == 0:
                                print(f"  Progress: {combo_count}/{total_combos} combinations tested...")
                            
                            try:
                                # Train Random Forest model with current hyperparameters
                                rf = RandomForestClassifier(
                                    n_estimators=n_est,
                                    max_depth=max_depth,
                                    min_samples_split=min_split,
                                    min_samples_leaf=min_leaf,
                                    max_features=max_feat,
                                    criterion=criterion,
                                    random_state=42,
                                    n_jobs=-1  # Use all CPU cores
                                )
                                rf.fit(X_train, y_train)
                                
                                # Evaluate on validation set
                                val_accuracy = rf.score(X_val, y_val)
                                
                                # Update best model if this is better
                                if val_accuracy > best_accuracy:
                                    best_accuracy = val_accuracy
                                    best_model = rf
                                    best_params = {
                                        'n_estimators': n_est,
                                        'max_depth': max_depth,
                                        'min_samples_split': min_split,
                                        'min_samples_leaf': min_leaf,
                                        'max_features': max_feat,
                                        'criterion': criterion,
                                        'accuracy': val_accuracy
                                    }
                                    print(f"  *** New best! Accuracy: {val_accuracy:.4f} ***")
                                    
                            except Exception as e:
                                # Handle cases where certain combinations might fail
                                print(f"Failed for combo {combo_count}: {e}")
                                continue
    
    # Print best configuration found
    if best_model is not None:
        print(f"\nBest Random Forest model found:")
        print(f"  n_estimators={best_params['n_estimators']}")
        print(f"  max_depth={best_params['max_depth']}")
        print(f"  min_samples_split={best_params['min_samples_split']}")
        print(f"  min_samples_leaf={best_params['min_samples_leaf']}")
        print(f"  max_features={best_params['max_features']}")
        print(f"  criterion={best_params['criterion']}")
        print(f"  validation accuracy={best_params['accuracy']:.4f}")
    else:
        print("No valid Random Forest model found!")
    
    return best_model

def best_multinomial_naive_bayes_model(X_train, y_train, X_val, y_val):
    """
    Find the best Naive Bayes model based on validation accuracy.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_val: Validation feature matrix
        y_val: Validation labels

    Returns:
        Trained Naive Bayes model with best hyperparameters
    """
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    fit_priors = [True, False]
    
    best_accuracy = 0
    best_model = None
    best_params = {}
    
    # Grid search over all hyperparameter combinations
    for alpha in alphas:
        for fit_prior in fit_priors:
            try:
                # Train Multinomial Naive Bayes model with current hyperparameters
                nb = MultinomialNB(
                    alpha=alpha,
                    fit_prior=fit_prior
                )
                nb.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_accuracy = nb.score(X_val, y_val)
                
                # Update best model if this is better
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_model = nb
                    best_params = {
                        'alpha': alpha,
                        'fit_prior': fit_prior,
                        'accuracy': val_accuracy
                    }
                    
            except Exception as e:
                # Handle cases where certain combinations might fail
                # (e.g., negative values in feature matrix)
                print(f"Failed for alpha={alpha}, fit_prior={fit_prior}: {e}")
                continue
    
    # Print best configuration found
    if best_model is not None:
        print(f"Best Naive Bayes model found:")
        print(f"  alpha={best_params['alpha']}")
        print(f"  fit_prior={best_params['fit_prior']}")
        print(f"  validation accuracy={best_params['accuracy']:.4f}")
    else:
        print("No valid Naive Bayes model found!")
    
    return best_model

"""preprocessing functions"""
def preprocess(df, max_features=50, 
               use_tfidf=True, 
               feature_combo = ('ratings', 'best_tasks', 'subopt_tasks', 'text'), 
               multiselect_tasks=all_multiselect_tasks,
               fitted_encoders=None):
    """
    Preprocess data with feature extraction.
    
    Args:
        df: DataFrame to process
        max_features: Max features for text vectorization
        use_tfidf: Whether to use TF-IDF
        feature_combo: Tuple of feature groups to include
        multiselect_tasks: List of multiselect task options
        fitted_encoders: Dict of fitted encoders (for validation/test sets)
                        If None, will fit new encoders (for training set)
    
    Returns:
        If fitted_encoders is None: (X, y, encoders)
        Otherwise: (X, y)
    """
    
    # Drop rows with missing data
    df = df.dropna()

    # Build feature matrix
    if fitted_encoders is None:
        # Training phase: fit encoders and return them
        X, encoders = build_features(
            df, 
            feature_combo, 
            multiselect_tasks=multiselect_tasks, 
            max_features=max_features, 
            use_tfidf=use_tfidf,
            fitted_encoders=None
        )
        y = df['label'].values
        return X, y, encoders
    else:
        # Validation/test phase: use fitted encoders
        X, _ = build_features(
            df, 
            feature_combo, 
            multiselect_tasks=multiselect_tasks, 
            max_features=max_features, 
            use_tfidf=use_tfidf,
            fitted_encoders=fitted_encoders
        )
        y = df['label'].values
        return X, y

def build_features(df, feature_combo, multiselect_tasks=all_multiselect_tasks, max_features=50, use_tfidf=False, fitted_encoders=None):
    """
    Build feature matrix based on specified feature combination.
    
    Args:
        df: DataFrame with data
        feature_combo: tuple of feature group names to include
        max_features: max features for text vectorization
        use_tfidf: whether to use TF-IDF
        multiselect_tasks: list of tasks for multi-select processing
        fitted_encoders: Dict of fitted encoders from training (None to fit new ones)
    
    Returns:
        If fitted_encoders is None: (Feature matrix, encoders dict)
        Otherwise: (Feature matrix, None)
    """
    features_list = []
    encoders = {} if fitted_encoders is None else fitted_encoders
    
    # Rating features (no encoding needed)
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
        
        if fitted_encoders is None:
            # Fit new encoder
            mlb_best = MultiLabelBinarizer()
            best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
            encoders['mlb_best'] = mlb_best
        else:
            # Use fitted encoder
            mlb_best = fitted_encoders['mlb_best']
            best_tasks_encoded = mlb_best.transform(best_tasks_lists)
        
        features_list.append(best_tasks_encoded)
    
    if 'subopt_tasks' in feature_combo:
        suboptimal_tasks_lists = process_multiselect(
            df['For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'], 
            multiselect_tasks
        )
        
        if fitted_encoders is None:
            # Fit new encoder
            mlb_subopt = MultiLabelBinarizer()
            suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)
            encoders['mlb_subopt'] = mlb_subopt
        else:
            # Use fitted encoder
            mlb_subopt = fitted_encoders['mlb_subopt']
            suboptimal_tasks_encoded = mlb_subopt.transform(suboptimal_tasks_lists)
        
        features_list.append(suboptimal_tasks_encoded)
    
    # Text features
    if 'text' in feature_combo:
        if fitted_encoders is None:
            text_features, text_encoders = one_hot_encode(df, max_features=max_features, use_tfidf=use_tfidf, fitted_vectorizers=None)
            encoders['text_vectorizers'] = text_encoders
        else:
            text_features, _ = one_hot_encode(df, max_features=max_features, use_tfidf=use_tfidf, fitted_vectorizers=fitted_encoders.get('text_vectorizers'))
        
        features_list.append(text_features)
    
    # Combine all selected features
    if features_list:
        X = np.hstack(features_list)
    else:
        X = np.array([]).reshape(len(df), 0)
    
    # Return encoders only if we fitted new ones
    return (X, encoders) if fitted_encoders is None else (X, None)

"""Helper functions"""

def one_hot_encode(df, max_features=50, use_tfidf=False, fitted_vectorizers=None):
    """
    Encode text columns using CountVectorizer or TfidfVectorizer.
    
    Args:
        df: DataFrame with text columns
        max_features: Max features for vectorization
        use_tfidf: Whether to use TF-IDF
        fitted_vectorizers: List of fitted vectorizers (None to fit new ones)
    
    Returns:
        If fitted_vectorizers is None: (encoded features, list of vectorizers)
        Otherwise: (encoded features, None)
    """
    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer

    text_columns = [
        "In your own words, what kinds of tasks would you use this model for?",
        "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
        "When you verify a response from this model, how do you usually go about it?"
    ]
    
    encoded_features = []
    vectorizers = [] if fitted_vectorizers is None else fitted_vectorizers
    
    for i, col in enumerate(text_columns):
        if col in df.columns:
            text_data = df[col].fillna('')
            
            if fitted_vectorizers is None:
                # Fit new vectorizer
                vectorizer = Vectorizer(
                    max_features=max_features,
                    stop_words='english',
                    min_df=2,
                    lowercase=True,
                    token_pattern=r'\b[a-zA-Z]{3,}\b'
                )
                features = vectorizer.fit_transform(text_data)
                vectorizers.append(vectorizer)
            else:
                # Use fitted vectorizer
                vectorizer = fitted_vectorizers[i]
                features = vectorizer.transform(text_data)
            
            encoded_features.append(features.toarray())
    
    if encoded_features:
        combined = np.hstack(encoded_features)
    else:
        combined = np.array([]).reshape(len(df), 0)
    
    # Return vectorizers only if we fitted new ones
    return (combined, vectorizers) if fitted_vectorizers is None else (combined, None)
    
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

def split(df, train_ratio=0.8, val_ratio=0.05, random_state=42):
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

def feature_combinations():
    """
    Return all non-empty combinations of feature groups.
    
    Returns:
        list of tuples: All possible combinations of feature groups
    """
    # Feature groups to test
    feature_groups = ['ratings', 'best_tasks', 'subopt_tasks', 'text']
    
    # Generate all non-empty combinations of feature groups
    all_combos = []
    for r in range(1, len(feature_groups) + 1):
        all_combos.extend(combinations(feature_groups, r))
    
    return all_combos

def calculate_f1_score(y_true, y_pred, average='binary', pos_label=1):
    """Calculate F1 score."""
    return f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)


def calculate_precision(y_true, y_pred, average='binary', pos_label=1):
    """Calculate precision score."""
    return precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, 
                         title='Confusion Matrix', cmap='Blues', 
                         figsize=(8, 6), save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        cm_display = cm * 100
    else:
        fmt = 'd'
        cm_display = cm
    
    if labels is None:
        labels = sorted(np.unique(y_true))
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Percentage' if normalize else 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax, cm

def main_comprehensive():
    """
    Alternative main function that tests ALL feature combinations
    to find the absolute best Naive Bayes model configuration.
    This is more thorough but takes longer to run.
    """
    # Load processed data
    df = pd.read_csv(file_name)
    
    # Split into train/val/test datasets
    train_df, val_df, test_df = split(df)
    
    # Find best model across all feature combinations
    result = find_best_model(
        train_df, 
        val_df, 
        model_family='random_forest',
        max_features=50,
        use_tfidf=False
    )
    
    if result['model'] is not None:
        # Get the best feature combo and retrain on combined train+val
        best_combo = result['feature_combo']
        
        # Preprocess with best feature combination
        X_train, y_train, encoders = preprocess(
            train_df,
            feature_combo=best_combo,
            fitted_encoders=None
        )
        X_test, y_test = preprocess(
            test_df,
            feature_combo=best_combo,
            fitted_encoders=encoders
        )
        
        # Evaluate best model on test set
        test_accuracy = result['model'].score(X_test, y_test)
        
        print(f"\n{'='*80}")
        print(f"FINAL TEST SET RESULTS")
        print(f"{'='*80}")
        print(f"Best feature combination: {best_combo}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"{'='*80}\n")
    else:
        print("No valid model found!")

    

if __name__ == "__main__":
    # Option 1: Run standard main (faster, single feature combo)
    #main()
    
    # Option 2: Run comprehensive search (slower, tests all feature combos)
    # Uncomment the line below to test all feature combinations:
    main_comprehensive()