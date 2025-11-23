"""
Enhanced kNN Model Finder with Feature Selection
This script implements feature selection to reduce overfitting on 800 training samples.
"""

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

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

"""Model testing functions with cross-validation"""
def best_knn_model_cv(X_train, y_train, X_val, y_val, cv_folds=5):
    """
    Find the best k for kNN based on cross-validation and validation accuracy.
    Uses larger k values to reduce overfitting.
    """
    # LARGER k values to reduce overfitting on 800 samples
    k_values = [11, 15, 21, 31, 41, 51]
    distances = ['euclidean']
    weights = ['uniform', 'distance']
    
    best_cv_score = 0
    best_model = None
    best_params = {}
    
    # Use stratified k-fold for cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    print(f"    Testing {len(k_values)} k values × {len(distances)} distances × {len(weights)} weights...")
    
    # Grid search over all hyperparameter combinations
    for k in k_values:
        for distance in distances:
            for weight in weights:
                try:
                    # Create kNN model with current hyperparameters
                    knn = KNeighborsClassifier(
                        n_neighbors=k,
                        metric=distance,
                        weights=weight
                    )
                    
                    # Perform cross-validation
                    cv_scores = cross_val_score(
                        knn, X_train, y_train, 
                        cv=skf, 
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    # Update best model if this is better
                    if cv_mean > best_cv_score:
                        # Train on full training set
                        knn.fit(X_train, y_train)
                        
                        # Evaluate on both training and validation sets
                        train_accuracy = knn.score(X_train, y_train)
                        val_accuracy = knn.score(X_val, y_val)
                        
                        best_cv_score = cv_mean
                        best_model = knn
                        best_params = {
                            'k': k,
                            'distance': distance,
                            'weight': weight,
                            'cv_mean': cv_mean,
                            'cv_std': cv_std,
                            'train_accuracy': train_accuracy,
                            'val_accuracy': val_accuracy
                        }
                        
                except Exception as e:
                    # Handle cases where certain combinations might fail
                    continue
    
    # Print best configuration found
    if best_model is not None:
        print(f"    Best kNN model found:")
        print(f"      k={best_params['k']}, distance={best_params['distance']}, weight={best_params['weight']}")
        print(f"      CV accuracy: {best_params['cv_mean']:.4f} (±{best_params['cv_std']:.4f})")
        print(f"      Train accuracy: {best_params['train_accuracy']:.4f}")
        print(f"      Val accuracy: {best_params['val_accuracy']:.4f}")
        
        # Calculate and display overfitting gap
        overfit_gap = best_params['train_accuracy'] - best_params['val_accuracy']
        print(f"      Overfitting gap: {overfit_gap:.4f}")
    else:
        print("    No valid kNN model found!")
    
    return best_model, best_params

"""Feature Selection Functions"""
def select_features(X_train, y_train, X_val, X_test, n_features, method='f_classif'):
    """
    Select top k features using specified method.
    
    Args:
        X_train, y_train: Training data
        X_val, X_test: Validation and test data (to transform with same selector)
        n_features: Number of features to select
        method: 'f_classif', 'mutual_info', or 'chi2'
    
    Returns:
        Transformed X_train, X_val, X_test, and the selector
    """
    # Choose scoring function
    if method == 'f_classif':
        score_func = f_classif
    elif method == 'mutual_info':
        score_func = mutual_info_classif
    elif method == 'chi2':
        score_func = chi2
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create and fit selector
    selector = SelectKBest(score_func=score_func, k=min(n_features, X_train.shape[1]))
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_val_selected, X_test_selected, selector

"""Preprocessing functions"""
def preprocess(df, max_features=20,  # REDUCED from 50 to 20
               use_tfidf=True, 
               feature_combo = ('ratings', 'best_tasks', 'subopt_tasks', 'text'), 
               multiselect_tasks=all_multiselect_tasks,
               fitted_encoders=None):
    """
    Preprocess data with feature extraction.
    Using max_features=20 instead of 50 to reduce dimensionality.
    """
    df = df.dropna()

    if fitted_encoders is None:
        X, encoders = build_features(
            df, feature_combo, multiselect_tasks=multiselect_tasks, 
            max_features=max_features, use_tfidf=use_tfidf, fitted_encoders=None
        )
        y = df['label'].values
        return X, y, encoders
    else:
        X, _ = build_features(
            df, feature_combo, multiselect_tasks=multiselect_tasks, 
            max_features=max_features, use_tfidf=use_tfidf, fitted_encoders=fitted_encoders
        )
        y = df['label'].values
        return X, y

def build_features(df, feature_combo, multiselect_tasks=all_multiselect_tasks, 
                   max_features=20, use_tfidf=True, fitted_encoders=None):
    """Build feature matrix based on specified feature combination."""
    features_list = []
    encoders = {} if fitted_encoders is None else fitted_encoders
    
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
        
        if fitted_encoders is None:
            mlb_best = MultiLabelBinarizer()
            best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
            encoders['mlb_best'] = mlb_best
        else:
            mlb_best = fitted_encoders['mlb_best']
            best_tasks_encoded = mlb_best.transform(best_tasks_lists)
        
        features_list.append(best_tasks_encoded)
    
    if 'subopt_tasks' in feature_combo:
        suboptimal_tasks_lists = process_multiselect(
            df['For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'], 
            multiselect_tasks
        )
        
        if fitted_encoders is None:
            mlb_subopt = MultiLabelBinarizer()
            suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)
            encoders['mlb_subopt'] = mlb_subopt
        else:
            mlb_subopt = fitted_encoders['mlb_subopt']
            suboptimal_tasks_encoded = mlb_subopt.transform(suboptimal_tasks_lists)
        
        features_list.append(suboptimal_tasks_encoded)
    
    # Text features
    if 'text' in feature_combo:
        if fitted_encoders is None:
            text_features, text_encoders = one_hot_encode(df, max_features=max_features, 
                                                         use_tfidf=use_tfidf, fitted_vectorizers=None)
            encoders['text_vectorizers'] = text_encoders
        else:
            text_features, _ = one_hot_encode(df, max_features=max_features, use_tfidf=use_tfidf, 
                                             fitted_vectorizers=fitted_encoders.get('text_vectorizers'))
        
        features_list.append(text_features)
    
    # Combine all selected features
    if features_list:
        X = np.hstack(features_list)
    else:
        X = np.array([]).reshape(len(df), 0)
    
    return (X, encoders) if fitted_encoders is None else (X, None)

"""Helper functions"""
def one_hot_encode(df, max_features=20, use_tfidf=True, fitted_vectorizers=None):
    """Encode text columns using CountVectorizer or TfidfVectorizer."""
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
                vectorizer = fitted_vectorizers[i]
                features = vectorizer.transform(text_data)
            
            encoded_features.append(features.toarray())
    
    if encoded_features:
        combined = np.hstack(encoded_features)
    else:
        combined = np.array([]).reshape(len(df), 0)
    
    return (combined, vectorizers) if fitted_vectorizers is None else (combined, None)
    
def process_multiselect(series, multiselect_tasks):
    """Convert multiselect strings to lists, keeping only specified features"""
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            present_tasks = [task for task in multiselect_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed

def extract_rating(response):
    """Extract numeric rating from responses like '3 - Sometimes'."""
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None

def split(df, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """Split data by student_id to prevent leakage."""
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
    
    return train_df, val_df, test_df

def feature_combinations():
    """Return all non-empty combinations of feature groups."""
    feature_groups = ['ratings', 'best_tasks', 'subopt_tasks']
    
    all_combos = []
    for r in range(1, len(feature_groups) + 1):
        all_combos.extend(combinations(feature_groups, r))
    
    return all_combos

def calculate_f1_score(y_true, y_pred, pos_label=1):
    """Calculate F1 score, automatically handling binary or multiclass."""
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(unique_labels) > 2:
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        return f1_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0)

def calculate_precision(y_true, y_pred, pos_label=1):
    """Calculate precision score, automatically handling binary or multiclass."""
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(unique_labels) > 2:
        return precision_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        return precision_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0)

def main_with_feature_selection():
    """
    Main function that tests feature combinations WITH feature selection
    to combat overfitting on 800 training samples.
    """
    print("="*80)
    print("kNN MODEL SEARCH WITH FEATURE SELECTION (800 samples)")
    print("="*80)
    
    # Load processed data
    print("\n1. Loading data...")
    df = pd.read_csv(file_name)
    print(f"   Loaded {len(df)} samples")
    print(f"   Label distribution: {dict(df['label'].value_counts())}")
    
    # Split into train/val/test datasets
    print("\n2. Splitting data by student_id...")
    train_df, val_df, test_df = split(df)
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    # Get all feature combinations to test
    all_combos = feature_combinations()
    print(f"\n3. Testing {len(all_combos)} feature combinations with feature selection...")
    
    # Feature selection parameters
    n_features_to_test = [10, 15, 20, 30, 40, 50]  # Different numbers of features to keep
    selection_methods = ['f_classif', 'mutual_info']  # Feature selection methods
    
    # Track best overall model
    best_overall_score = 0
    best_overall_model = None
    best_overall_config = {}
    results = []
    
    # Test each feature combination
    for i, combo in enumerate(all_combos, 1):
        print(f"\n{'='*80}")
        print(f"Combination {i}/{len(all_combos)}: {combo}")
        print(f"{'='*80}")
        
        try:
            # Preprocess training data (fit encoders)
            X_train_full, y_train, encoders = preprocess(
                train_df, 
                max_features=20,  # Reduced from 50
                use_tfidf=True,
                feature_combo=combo,
                fitted_encoders=None
            )
            
            # Preprocess validation and test data (use fitted encoders)
            X_val_full, y_val = preprocess(
                val_df,
                max_features=20,
                use_tfidf=True,
                feature_combo=combo,
                fitted_encoders=encoders
            )
            
            X_test_full, y_test = preprocess(
                test_df,
                max_features=20,
                use_tfidf=True,
                feature_combo=combo,
                fitted_encoders=encoders
            )
            
            print(f"Initial feature matrix shape: {X_train_full.shape}")
            
            # Try different feature selection configurations
            for method in selection_methods:
                for n_features in n_features_to_test:
                    if n_features >= X_train_full.shape[1]:
                        # Skip if we're trying to select more features than we have
                        continue
                    
                    print(f"\n--- Method: {method}, Features: {n_features} ---")
                    
                    # Apply feature selection
                    X_train, X_val, X_test, selector = select_features(
                        X_train_full, y_train, X_val_full, X_test_full,
                        n_features=n_features,
                        method=method
                    )
                    
                    print(f"  Selected feature matrix shape: {X_train.shape}")
                    
                    # Find best kNN model for this configuration
                    model, params = best_knn_model_cv(X_train, y_train, X_val, y_val, cv_folds=5)
                    
                    if model is not None:
                        # Calculate training metrics
                        y_train_pred = model.predict(X_train)
                        train_acc = accuracy_score(y_train, y_train_pred)
                        train_f1 = calculate_f1_score(y_train, y_train_pred)
                        
                        # Calculate validation metrics
                        y_val_pred = model.predict(X_val)
                        val_acc = accuracy_score(y_val, y_val_pred)
                        val_f1 = calculate_f1_score(y_val, y_val_pred)
                        
                        # Calculate test metrics
                        y_test_pred = model.predict(X_test)
                        test_acc = accuracy_score(y_test, y_test_pred)
                        test_f1 = calculate_f1_score(y_test, y_test_pred)
                        
                        # Calculate overfitting indicators
                        overfit_gap = train_acc - val_acc
                        
                        print(f"  Performance:")
                        print(f"    Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
                        print(f"    Overfit gap: {overfit_gap:.4f}")
                        
                        # Store results
                        result = {
                            'combo': combo,
                            'method': method,
                            'n_features': n_features,
                            'initial_features': X_train_full.shape[1],
                            'cv_mean': params['cv_mean'],
                            'cv_std': params['cv_std'],
                            'train_acc': train_acc,
                            'val_acc': val_acc,
                            'test_acc': test_acc,
                            'train_f1': train_f1,
                            'val_f1': val_f1,
                            'test_f1': test_f1,
                            'overfit_gap': overfit_gap,
                            'k': params['k'],
                            'distance': params['distance'],
                            'weight': params['weight']
                        }
                        results.append(result)
                        
                        # Update best overall model based on validation accuracy
                        # and considering overfitting gap
                        if val_acc > best_overall_score and overfit_gap < 0.15:
                            best_overall_score = val_acc
                            best_overall_model = model
                            best_overall_config = {
                                'combo': combo,
                                'encoders': encoders,
                                'selector': selector,
                                'method': method,
                                'n_features': n_features,
                                'params': params,
                                'train_acc': train_acc,
                                'val_acc': val_acc,
                                'test_acc': test_acc,
                                'train_f1': train_f1,
                                'val_f1': val_f1,
                                'test_f1': test_f1,
                                'overfit_gap': overfit_gap
                            }
                            print(f"  *** NEW BEST MODEL (Val Acc: {val_acc:.4f}) ***")
                        
        except Exception as e:
            print(f"  Error with combination {combo}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary of results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Sort by validation accuracy (prioritizing generalization)
    results_sorted = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    
    print(f"\nTop 15 models by VALIDATION accuracy:")
    print(f"{'Rank':<5} {'Features':<25} {'Method':<12} {'#Feat':<7} {'Train':<8} {'Val':<8} {'Test':<8} {'Gap':<7} {'k':<5}")
    print("-"*110)
    
    for rank, r in enumerate(results_sorted[:15], 1):
        combo_str = '+'.join(r['combo'])
        print(f"{rank:<5} {combo_str:<25} {r['method']:<12} {r['n_features']:<7} {r['train_acc']:.4f}   {r['val_acc']:.4f}   {r['test_acc']:.4f}   {r['overfit_gap']:.4f}  {r['k']:<5}")
    
    # Also show models with smallest overfitting gap
    print(f"\n\nTop 10 models by SMALLEST overfitting gap:")
    results_by_gap = sorted(results, key=lambda x: x['overfit_gap'])
    
    print(f"{'Rank':<5} {'Features':<25} {'Method':<12} {'#Feat':<7} {'Train':<8} {'Val':<8} {'Test':<8} {'Gap':<7} {'k':<5}")
    print("-"*110)
    
    for rank, r in enumerate(results_by_gap[:10], 1):
        combo_str = '+'.join(r['combo'])
        print(f"{rank:<5} {combo_str:<25} {r['method']:<12} {r['n_features']:<7} {r['train_acc']:.4f}   {r['val_acc']:.4f}   {r['test_acc']:.4f}   {r['overfit_gap']:.4f}  {r['k']:<5}")
    
    # Evaluate best model
    if best_overall_model is not None:
        print("\n" + "="*80)
        print("BEST MODEL (Selected by Val Accuracy with Overfitting < 15%)")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  Features: {'+'.join(best_overall_config['combo'])}")
        print(f"  Selection method: {best_overall_config['method']}")
        print(f"  Number of features: {best_overall_config['n_features']}")
        print(f"  k={best_overall_config['params']['k']}")
        print(f"  distance={best_overall_config['params']['distance']}")
        print(f"  weight={best_overall_config['params']['weight']}")
        
        print(f"\n{'Set':<15} {'Accuracy':<12} {'F1 Score':<12}")
        print("-"*40)
        print(f"{'Training':<15} {best_overall_config['train_acc']:<12.4f} {best_overall_config['train_f1']:<12.4f}")
        print(f"{'Validation':<15} {best_overall_config['val_acc']:<12.4f} {best_overall_config['val_f1']:<12.4f}")
        print(f"{'Test':<15} {best_overall_config['test_acc']:<12.4f} {best_overall_config['test_f1']:<12.4f}")
        
        print(f"\nOverfitting Analysis:")
        print(f"  Train-Val gap: {best_overall_config['train_acc'] - best_overall_config['val_acc']:.4f}")
        print(f"  Train-Test gap: {best_overall_config['train_acc'] - best_overall_config['test_acc']:.4f}")
        print(f"  Val-Test gap: {best_overall_config['val_acc'] - best_overall_config['test_acc']:.4f}")
        
        print("\n" + "="*80)
        print("SEARCH COMPLETE!")
        print("="*80)
        
        return best_overall_model, best_overall_config, results_sorted
    else:
        print("\nNo valid model found!")
        return None, None, results_sorted

if __name__ == "__main__":
    # Run search with feature selection to combat overfitting
    best_model, best_config, all_results = main_with_feature_selection()