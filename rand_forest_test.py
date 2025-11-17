from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import joblib
import data_encoding as de
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TrainingForest:
    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf):
        # self.data_path = data_path
        # self.data_path_train = data_path_train
        # self.data_path_valid = data_path_valid
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',        # Add: limits features per split (reduces overfitting)
            min_impurity_decrease=0.0,  # Add: can increase to 0.01 for more regularization
            bootstrap=True,             # Already default, but explicit
            oob_score=True,            # Add: provides out-of-bag validation score
            n_jobs=-1,
            random_state=42
        )
        # self.model = MultiOutputRegressor(base)
        self.X_train, self.X_valid, self.y_train, self.y_valid = None, None, None, None

    def load_data(self, X_train, y_train, X_valid, y_valid):
        # data_train = pd.read_csv(self.data_path_train)
        # data_valid = pd.read_csv(self.data_path_valid)
        # X_train = data_train.drop(['ChatGPT', 'Claude', 'Gemini'], axis=1)     # change to label of targets.
        # y_train = data_train[['ChatGPT', 'Claude', 'Gemini']]
        # X_valid = data_valid.drop(['ChatGPT', 'Claude', 'Gemini'], axis=1)
        # y_valid = data_valid[['ChatGPT', 'Claude', 'Gemini']]
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid

    # def encoder():
    #     encoder = TargetEncoder(cols=self.X_train.columns.tolist(), smoothing=0.3)  # smoothing reduces overfitting on rare genes
    #     X_train_encoded = encoder.fit_transform(X_train, y_train)
    #     X_test_encoded = encoder.transform(X_test)
    #     self.y_train = label_encoder.fit_transform(y_train)
    #     self.y_valid = label_encoder.transform(y_test)
    #     num_cols = [c for c in X_train_encoded.columns if c not in self.X_train.columns.tolist()]
    #     scaler = StandardScaler()
    #     if num_cols:
    #         X_train_encoded[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
    #         X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])
    #     self.X_train = X_train_encoded
    #     self.X_test = X_test_encoded


    def train_model(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() before training the model.")
        self.model.fit(self.X_train, self.y_train)

    # use the universal code.
    def evaluate_model(self):
        if self.X_valid is None or self.y_valid is None:
            raise ValueError("Data not loaded. Call load_data() before evaluating the model.")
        predictions = self.model.predict(self.X_valid)
        
        # Calculate metrics using sklearn (handles multi-class automatically)
        accuracy = accuracy_score(self.y_valid, predictions)
        precision = precision_score(self.y_valid, predictions, average='weighted', zero_division=0)
        recall = recall_score(self.y_valid, predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.y_valid, predictions, average='weighted', zero_division=0)
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
    
    def evaluate_train(self):
        """Evaluate model performance on training data"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() before evaluating the model.")
        predictions = self.model.predict(self.X_train)
        
        # Calculate metrics on training data
        accuracy = accuracy_score(self.y_train, predictions)
        precision = precision_score(self.y_train, predictions, average='weighted', zero_division=0)
        recall = recall_score(self.y_train, predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.y_train, predictions, average='weighted', zero_division=0)
        
        return {"train_accuracy": accuracy, "train_precision": precision, "train_recall": recall, "train_f1_score": f1}

    # def update_hyperparameters(self, n_estimators, max_depth, min_samples_split=2, min_samples_leaf=1):
    #     """Update model with new hyperparameters"""
    #     base = RandomForestClassifier(
    #         n_estimators=n_estimators,
    #         max_depth=max_depth,
    #         min_samples_split=min_samples_split,
    #         min_samples_leaf=min_samples_leaf,
    #         n_jobs=-1,
    #         random_state=42
    #     )
    #     self.model = MultiOutputRegressor(base)

def hyperparameter_search(X_train, y_train, X_valid, y_valid):
    """Test different hyperparameter combinations for RandomForestClassifier"""
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [8, 16, 32, 64, 100, 128, 200, 300],
        'max_depth': [6, 8, 10, 12, 14, 16, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8, 12]
    }
    
    results = []
    best_f1 = -float('inf')
    best_params = None
    
    print("Starting hyperparameter search...")
    print(f"Total combinations to test: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])}")
    print("-" * 80)
    
    iteration = 0
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                for min_leaf in param_grid['min_samples_leaf']:
                    iteration += 1
                    print(f"\nIteration {iteration}: n_estimators={n_est}, max_depth={max_d}, min_samples_split={min_split}, min_samples_leaf={min_leaf}")
                    
                    # Create and train model
                    model = TrainingForest(n_est, max_d, min_split, min_leaf)
                    model.load_data(X_train, y_train, X_valid, y_valid)
                    # model.update_hyperparameters(n_est, max_d, min_split, min_leaf)
                    model.train_model()
                    
                    # Evaluate
                    metrics = model.evaluate_model()
                    
                    # Store results
                    result = {
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_split': min_split,
                        'min_samples_leaf': min_leaf,
                        **metrics  # Unpack all metrics returned
                    }
                    results.append(result)
                    
                    print(f"  " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
                    
                    # Track best model by f1_score
                    if metrics['f1_score'] > best_f1:
                        best_f1 = metrics['f1_score']
                        best_params = result.copy()
                        print(f"  *** New best F1 score! ***")
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nBest Parameters:")
    print(f"  n_estimators: {best_params['n_estimators']}")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  min_samples_split: {best_params['min_samples_split']}")
    print(f"  min_samples_leaf: {best_params['min_samples_leaf']}")
    print(f"\nBest Metrics:")
    for k, v in best_params.items():
        if k not in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
            print(f"  {k}: {v:.4f}")
    
    # Save results to CSV (sorted by f1_score)
    results_df = pd.DataFrame(results)
    sort_key = 'f1_score'
    results_df = results_df.sort_values(sort_key, ascending=False)
    results_df.to_csv('hyperparameter_search_results.csv', index=False)
    print(f"\nAll results saved to 'hyperparameter_search_results.csv'")
    
    # Train best model and get training metrics for report
    print("\nEvaluating best model on training data...")
    best_model = TrainingForest(
        best_params['n_estimators'],
        best_params['max_depth'],
        best_params['min_samples_split'],
        best_params['min_samples_leaf']
    )
    best_model.load_data(X_train, y_train, X_valid, y_valid)
    best_model.train_model()
    train_metrics = best_model.evaluate_train()
    
    # Save formatted text report
    with open('hyperparameter_search_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RANDOM FOREST HYPERPARAMETER SEARCH REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Combinations Tested: {len(results)}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("BEST MODEL\n")
        f.write("-" * 80 + "\n")
        f.write(f"n_estimators:      {best_params['n_estimators']}\n")
        f.write(f"max_depth:         {best_params['max_depth']}\n")
        f.write(f"min_samples_split: {best_params['min_samples_split']}\n")
        f.write(f"min_samples_leaf:  {best_params['min_samples_leaf']}\n\n")
        f.write(f"Validation Performance:\n")
        for k, v in best_params.items():
            if k not in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                f.write(f"  {k}: {v:.6f}\n")
        f.write("\n")
        f.write(f"Training Performance:\n")
        for k, v in train_metrics.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n")
        f.write(f"Overfitting Analysis:\n")
        f.write(f"  Accuracy gap (train - val): {train_metrics['train_accuracy'] - best_params['accuracy']:.6f}\n")
        f.write(f"  F1 gap (train - val): {train_metrics['train_f1_score'] - best_params['f1_score']:.6f}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"TOP 10 CONFIGURATIONS (by {sort_key})\n")
        f.write("-" * 80 + "\n\n")
        
        top_10 = results_df.head(10)
        for idx, row in top_10.iterrows():
            rank = list(top_10.index).index(idx) + 1
            f.write(f"Rank {rank}:\n")
            f.write(f"  n_estimators={row['n_estimators']}, max_depth={row['max_depth']}, min_samples_split={row['min_samples_split']}, min_samples_leaf={row['min_samples_leaf']}\n")
            for k in metrics.keys():
                f.write(f"  {k}: {row[k]:.6f}\n")
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("ALL RESULTS (sorted by F1 Score)\n")
        f.write("-" * 80 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Formatted report saved to 'hyperparameter_search_report.txt'")
    
    # Train final model with best params
    print("\nTraining final model with best parameters...")
    final_model = TrainingForest(
        best_params['n_estimators'],
        best_params['max_depth'],
        best_params['min_samples_split'],
        best_params['min_samples_leaf']
    )
    final_model.load_data(X_train, y_train, X_valid, y_valid)
    final_model.train_model()
    joblib.dump(final_model.model, 'best_forestclass_model.joblib')
    print("Best model saved to 'best_forestclass_model.joblib'")
    
    return best_params, results_df


def train(data_path_train, data_path_valid):
    """Train a single model with default parameters"""
    # model = TrainingForest(data_path_train, data_path_valid)
    # model.load_data()
    df_train = pd.read_csv('train.csv')
    df_valid = pd.read_csv('val.csv')
    X_train, y_train = de.preprocess(df_train)
    X_valid, y_valid = de.preprocess(df_valid)
    hyperparameter_search(X_train, y_train, X_valid, y_valid)
    # model.train_model(X_train, y_train)
    # print(model.evaluate_model())
    # joblib.dump(model.model, 'forestclass_model.joblib')

# if __name__ == "__main__":
#     # Example usage:
#     # Specify your data paths
#     train_path = 'path/to/train.csv'  # Replace with your training data path
#     valid_path = 'path/to/valid.csv'  # Replace with your validation data path
    
#     # Option 1: Run hyperparameter search
#     print("Running hyperparameter search...")
#     best_params, results = hyperparameter_search(train_path, valid_path)
    
#     # Option 2: Train single model with specific parameters (comment out if running search)
#     # train(train_path, valid_path)

train('train.csv', 'valid.csv')
# import time
# start = time.time()
# model = TrainingForest(64, 12, 2, 2)
# df_train = pd.read_csv('train.csv')
# df_valid = pd.read_csv('val.csv')
# X_train, y_train = de.preprocess(df_train)
# X_valid, y_valid = de.preprocess(df_valid)
# model.load_data(X_train, y_train, X_valid, y_valid)
# model.train_model()
# print("Single fit time:", time.time() - start, "seconds")
# print(model.evaluate_model())