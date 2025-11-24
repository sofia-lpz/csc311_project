"""
Comprehensive Cross-Validation Analysis
Tests the best model configuration with extensive cross-validation
to assess stability and get unbiased performance estimates.

FIXED: Now properly groups by student_id to prevent data leakage.
"""

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
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

# Best configuration found
BEST_CONFIG = {
    'feature_combo': ('ratings', 'best_tasks', 'subopt_tasks'),
    'n_features': 10,
    'k': 21,
    'distance': 'euclidean',
    'weight': 'uniform',
    'selection_method': 'f_classif'
}

"""Preprocessing functions"""
def preprocess(df, max_features=20, 
               use_tfidf=True, 
               feature_combo = ('ratings', 'best_tasks', 'subopt_tasks'), 
               multiselect_tasks=all_multiselect_tasks,
               fitted_encoders=None):
    """Preprocess data with feature extraction."""
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
    
    # Combine all selected features
    if features_list:
        X = np.hstack(features_list)
    else:
        X = np.array([]).reshape(len(df), 0)
    
    return (X, encoders) if fitted_encoders is None else (X, None)

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

def create_student_cv_folds(df, n_folds=10, random_state=42):
    """
    Create cross-validation folds that keep all responses from the same student together.
    
    Args:
        df: DataFrame with 'student_id' and 'label' columns
        n_folds: Number of folds
        random_state: Random seed
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Get unique students and their labels
    student_labels = df.groupby('student_id')['label'].first()
    unique_students = student_labels.index.values
    student_label_map = student_labels.to_dict()
    
    # Shuffle students
    np.random.seed(random_state)
    shuffled_students = np.random.permutation(unique_students)
    
    # Try to balance labels across folds (approximate stratification by student)
    # Group students by their label
    label_to_students = {}
    for student in shuffled_students:
        label = student_label_map[student]
        if label not in label_to_students:
            label_to_students[label] = []
        label_to_students[label].append(student)
    
    # Distribute students across folds trying to balance labels
    folds = [[] for _ in range(n_folds)]
    for label, students in label_to_students.items():
        for i, student in enumerate(students):
            folds[i % n_folds].append(student)
    
    # Create train/val splits
    cv_splits = []
    for val_fold_idx in range(n_folds):
        val_students = set(folds[val_fold_idx])
        train_students = set()
        for train_fold_idx in range(n_folds):
            if train_fold_idx != val_fold_idx:
                train_students.update(folds[train_fold_idx])
        
        # Get row indices for these students
        train_indices = df[df['student_id'].isin(train_students)].index.values
        val_indices = df[df['student_id'].isin(val_students)].index.values
        
        cv_splits.append((train_indices, val_indices))
    
    return cv_splits

def detailed_cross_validation(X, y, df, config, n_folds=10):
    """
    Perform detailed k-fold cross-validation with per-fold reporting.
    FIXED: Now splits by student_id to prevent data leakage.
    
    Args:
        X: Feature matrix
        y: Labels
        df: Original dataframe (needed for student_id)
        config: Model configuration dictionary
        n_folds: Number of CV folds
    
    Returns:
        Dictionary with detailed CV results
    """
    print(f"\n{'='*80}")
    print(f"DETAILED {n_folds}-FOLD CROSS-VALIDATION (Student-Grouped)")
    print(f"{'='*80}")
    
    # Create the model
    model = KNeighborsClassifier(
        n_neighbors=config['k'],
        metric=config['distance'],
        weights=config['weight']
    )
    
    # Create student-grouped k-fold splits
    cv_splits = create_student_cv_folds(df, n_folds=n_folds, random_state=42)
    
    # Store results for each fold
    fold_results = []
    all_y_true = []
    all_y_pred = []
    
    print(f"\nRunning {n_folds}-fold cross-validation (grouping by student_id)...")
    print(f"Note: Each student's 3 responses stay together in the same fold.")
    print(f"{'Fold':<6} {'Train Acc':<12} {'Val Acc':<12} {'Val F1':<12} {'Val Precision':<15} {'Val Recall':<12} {'Overfit Gap':<12}")
    print("-"*95)
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Apply feature selection
        selector = SelectKBest(score_func=f_classif, k=min(config['n_features'], X_train_fold.shape[1]))
        X_train_selected = selector.fit_transform(X_train_fold, y_train_fold)
        X_val_selected = selector.transform(X_val_fold)
        
        # Train model
        model.fit(X_train_selected, y_train_fold)
        
        # Predict
        y_train_pred = model.predict(X_train_selected)
        y_val_pred = model.predict(X_val_selected)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train_fold, y_train_pred)
        val_acc = accuracy_score(y_val_fold, y_val_pred)
        
        # Detect if binary or multiclass
        unique_labels = np.unique(y)
        avg_type = 'binary' if len(unique_labels) == 2 else 'macro'
        
        val_f1 = f1_score(y_val_fold, y_val_pred, average=avg_type, zero_division=0)
        val_precision = precision_score(y_val_fold, y_val_pred, average=avg_type, zero_division=0)
        val_recall = recall_score(y_val_fold, y_val_pred, average=avg_type, zero_division=0)
        overfit_gap = train_acc - val_acc
        
        # Store results
        fold_results.append({
            'fold': fold,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'overfit_gap': overfit_gap
        })
        
        # Store predictions for overall confusion matrix
        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_val_pred)
        
        # Print fold results
        print(f"{fold:<6} {train_acc:<12.4f} {val_acc:<12.4f} {val_f1:<12.4f} {val_precision:<15.4f} {val_recall:<12.4f} {overfit_gap:<12.4f}")
    
    print("-"*95)
    
    # Calculate statistics across folds
    train_accs = [r['train_acc'] for r in fold_results]
    val_accs = [r['val_acc'] for r in fold_results]
    val_f1s = [r['val_f1'] for r in fold_results]
    val_precisions = [r['val_precision'] for r in fold_results]
    val_recalls = [r['val_recall'] for r in fold_results]
    overfit_gaps = [r['overfit_gap'] for r in fold_results]
    
    # Print summary statistics
    print(f"\n{'SUMMARY STATISTICS'}")
    print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*70)
    print(f"{'Train Accuracy':<20} {np.mean(train_accs):<12.4f} {np.std(train_accs):<12.4f} {np.min(train_accs):<12.4f} {np.max(train_accs):<12.4f}")
    print(f"{'Val Accuracy':<20} {np.mean(val_accs):<12.4f} {np.std(val_accs):<12.4f} {np.min(val_accs):<12.4f} {np.max(val_accs):<12.4f}")
    print(f"{'Val F1 Score':<20} {np.mean(val_f1s):<12.4f} {np.std(val_f1s):<12.4f} {np.min(val_f1s):<12.4f} {np.max(val_f1s):<12.4f}")
    print(f"{'Val Precision':<20} {np.mean(val_precisions):<12.4f} {np.std(val_precisions):<12.4f} {np.min(val_precisions):<12.4f} {np.max(val_precisions):<12.4f}")
    print(f"{'Val Recall':<20} {np.mean(val_recalls):<12.4f} {np.std(val_recalls):<12.4f} {np.min(val_recalls):<12.4f} {np.max(val_recalls):<12.4f}")
    print(f"{'Overfitting Gap':<20} {np.mean(overfit_gaps):<12.4f} {np.std(overfit_gaps):<12.4f} {np.min(overfit_gaps):<12.4f} {np.max(overfit_gaps):<12.4f}")
    
    # Overall confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\nOverall Confusion Matrix (all folds combined):")
    print(cm)
    
    # Calculate overall metrics
    unique_labels = np.unique(y)
    avg_type = 'binary' if len(unique_labels) == 2 else 'macro'
    
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average=avg_type, zero_division=0)
    overall_precision = precision_score(all_y_true, all_y_pred, average=avg_type, zero_division=0)
    overall_recall = recall_score(all_y_true, all_y_pred, average=avg_type, zero_division=0)
    
    print(f"\nOverall Metrics (aggregated across all folds):")
    print(f"  Accuracy:  {overall_acc:.4f}")
    print(f"  F1 Score:  {overall_f1:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")
    
    return {
        'fold_results': fold_results,
        'mean_val_acc': np.mean(val_accs),
        'std_val_acc': np.std(val_accs),
        'mean_val_f1': np.mean(val_f1s),
        'std_val_f1': np.std(val_f1s),
        'mean_overfit_gap': np.mean(overfit_gaps),
        'std_overfit_gap': np.std(overfit_gaps),
        'overall_acc': overall_acc,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'confusion_matrix': cm
    }

def test_different_k_values(X, y, df, config, k_values, n_folds=10):
    """
    Test different k values with cross-validation to find optimal k.
    FIXED: Now splits by student_id to prevent data leakage.
    
    Args:
        X: Feature matrix
        y: Labels
        df: Original dataframe (needed for student_id)
        config: Base configuration (will vary k)
        k_values: List of k values to test
        n_folds: Number of CV folds
    """
    print(f"\n{'='*80}")
    print(f"TESTING DIFFERENT K VALUES (Student-Grouped CV)")
    print(f"{'='*80}")
    
    # Create student-grouped k-fold splits
    cv_splits = create_student_cv_folds(df, n_folds=n_folds, random_state=42)
    
    results = []
    
    print(f"\n{'k':<8} {'Mean Val Acc':<15} {'Std Val Acc':<15} {'Mean Overfit':<15} {'Std Overfit':<15}")
    print("-"*70)
    
    for k in k_values:
        # Create model with this k
        model = KNeighborsClassifier(
            n_neighbors=k,
            metric=config['distance'],
            weights=config['weight']
        )
        
        val_accs = []
        overfit_gaps = []
        
        for train_idx, val_idx in cv_splits:
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Apply feature selection
            selector = SelectKBest(score_func=f_classif, k=min(config['n_features'], X_train_fold.shape[1]))
            X_train_selected = selector.fit_transform(X_train_fold, y_train_fold)
            X_val_selected = selector.transform(X_val_fold)
            
            # Train and evaluate
            model.fit(X_train_selected, y_train_fold)
            
            train_acc = model.score(X_train_selected, y_train_fold)
            val_acc = model.score(X_val_selected, y_val_fold)
            
            val_accs.append(val_acc)
            overfit_gaps.append(train_acc - val_acc)
        
        mean_val_acc = np.mean(val_accs)
        std_val_acc = np.std(val_accs)
        mean_overfit = np.mean(overfit_gaps)
        std_overfit = np.std(overfit_gaps)
        
        results.append({
            'k': k,
            'mean_val_acc': mean_val_acc,
            'std_val_acc': std_val_acc,
            'mean_overfit': mean_overfit,
            'std_overfit': std_overfit
        })
        
        print(f"{k:<8} {mean_val_acc:<15.4f} {std_val_acc:<15.4f} {mean_overfit:<15.4f} {std_overfit:<15.4f}")
    
    # Find best k
    best_result = max(results, key=lambda x: x['mean_val_acc'])
    print(f"\nBest k: {best_result['k']} (Val Acc: {best_result['mean_val_acc']:.4f})")
    
    return results

def main_cv_analysis():
    """
    Main function to perform comprehensive cross-validation analysis.
    """
    print("="*80)
    print("COMPREHENSIVE CROSS-VALIDATION ANALYSIS")
    print("FIXED: Student-Grouped CV (No Data Leakage)")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(file_name)
    print(f"   Loaded {len(df)} samples")
    print(f"   Label distribution: {dict(df['label'].value_counts())}")
    
    # Check student distribution
    n_students = df['student_id'].nunique()
    samples_per_student = df.groupby('student_id').size()
    print(f"   Number of unique students: {n_students}")
    print(f"   Samples per student: {samples_per_student.value_counts().to_dict()}")
    
    # Split data (keep test set held out)
    print("\n2. Splitting data...")
    train_df, val_df, test_df = split(df)
    
    # Combine train and val for cross-validation
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"   Train+Val: {len(train_val_df)} samples (for CV)")
    print(f"   Train+Val students: {train_val_df['student_id'].nunique()}")
    print(f"   Test: {len(test_df)} samples (held out)")
    print(f"   Test students: {test_df['student_id'].nunique()}")
    
    # Preprocess train+val data
    print(f"\n3. Preprocessing data with config: {BEST_CONFIG['feature_combo']}...")
    # Clean the dataframe first (dropna) to match what preprocess does
    train_val_df_clean = train_val_df.dropna().reset_index(drop=True)
    
    X_train_val, y_train_val, encoders = preprocess(
        train_val_df_clean,
        max_features=20,
        use_tfidf=True,
        feature_combo=BEST_CONFIG['feature_combo'],
        fitted_encoders=None
    )
    print(f"   Feature matrix shape: {X_train_val.shape}")
    print(f"   Cleaned dataframe shape: {train_val_df_clean.shape}")
    
    # Detect classification type
    unique_labels = np.unique(y_train_val)
    print(f"   Number of classes: {len(unique_labels)}")
    print(f"   Class labels: {unique_labels}")
    print(f"   Classification type: {'Binary' if len(unique_labels) == 2 else 'Multiclass'}")
    
    # Perform detailed 10-fold cross-validation
    print(f"\n4. Running 10-fold cross-validation with best configuration...")
    cv_results = detailed_cross_validation(X_train_val, y_train_val, train_val_df_clean, BEST_CONFIG, n_folds=10)
    
    # Test different k values
    print(f"\n5. Testing different k values around optimal k={BEST_CONFIG['k']}...")
    k_values = [11, 15, 19, 21, 23, 25, 31, 41]
    k_results = test_different_k_values(X_train_val, y_train_val, train_val_df_clean, BEST_CONFIG, k_values, n_folds=10)
    
    # Final evaluation on held-out test set
    print(f"\n{'='*80}")
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print(f"{'='*80}")
    
    # Preprocess test data
    X_test, y_test = preprocess(
        test_df,
        max_features=20,
        use_tfidf=True,
        feature_combo=BEST_CONFIG['feature_combo'],
        fitted_encoders=encoders
    )
    
    # Apply feature selection
    selector = SelectKBest(score_func=f_classif, k=min(BEST_CONFIG['n_features'], X_train_val.shape[1]))
    X_train_val_selected = selector.fit_transform(X_train_val, y_train_val)
    X_test_selected = selector.transform(X_test)
    
    # Train final model on all train+val data
    final_model = KNeighborsClassifier(
        n_neighbors=BEST_CONFIG['k'],
        metric=BEST_CONFIG['distance'],
        weights=BEST_CONFIG['weight']
    )
    final_model.fit(X_train_val_selected, y_train_val)
    
    # Evaluate on test set
    y_train_val_pred = final_model.predict(X_train_val_selected)
    y_test_pred = final_model.predict(X_test_selected)
    
    # Detect if binary or multiclass
    unique_labels = np.unique(y_train_val)
    avg_type = 'binary' if len(unique_labels) == 2 else 'macro'
    
    train_val_acc = accuracy_score(y_train_val, y_train_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average=avg_type, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, average=avg_type, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average=avg_type, zero_division=0)
    
    print(f"\nFinal Model Performance:")
    print(f"  Train+Val Accuracy: {train_val_acc:.4f}")
    print(f"  Test Accuracy:      {test_acc:.4f}")
    print(f"  Test F1 Score:      {test_f1:.4f}")
    print(f"  Test Precision:     {test_precision:.4f}")
    print(f"  Test Recall:        {test_recall:.4f}")
    print(f"  Overfitting Gap:    {train_val_acc - test_acc:.4f}")
    
    cm_test = confusion_matrix(y_test, y_test_pred)
    print(f"\nTest Set Confusion Matrix:")
    print(cm_test)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nConfiguration: {BEST_CONFIG}")
    print(f"\n10-Fold CV Results (Student-Grouped):")
    print(f"  Mean Val Accuracy:  {cv_results['mean_val_acc']:.4f} ± {cv_results['std_val_acc']:.4f}")
    print(f"  Mean Val F1 Score:  {cv_results['mean_val_f1']:.4f} ± {cv_results['std_val_f1']:.4f}")
    print(f"  Mean Overfit Gap:   {cv_results['mean_overfit_gap']:.4f} ± {cv_results['std_overfit_gap']:.4f}")
    print(f"\nHeld-out Test Set:")
    print(f"  Test Accuracy:      {test_acc:.4f}")
    print(f"  Test F1 Score:      {test_f1:.4f}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print("No data leakage - students kept together in folds!")
    print(f"{'='*80}")
    
    return cv_results, k_results, final_model

if __name__ == "__main__":
    cv_results, k_results, model = main_cv_analysis()