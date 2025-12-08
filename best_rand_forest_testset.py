# Run the best random forest model on the test set
import pandas as pd
from sklearn.ensemble import RandomForestClassifier        
from sklearn.metrics import classification_report, accuracy_score
from data_encoding import preprocess, one_hot_encode
import data_encoding as de
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import json
from encoding import save_text_vocab  # legacy; we'll replace with exact sklearn vocab export



def data():
    """Train a single model with default parameters"""
    # model = TrainingForest(data_path_train, data_path_valid)
    # model.load_data()
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    X_train, y_train = de.preprocess(df_train)
    X_test, y_test = de.preprocess(df_test)
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=64,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=12,
            max_features='sqrt',        # Add: limits features per split (reduces overfitting)
            min_impurity_decrease=0.0,  # Add: can increase to 0.01 for more regularization
            bootstrap=True,             # Already default, but explicit
            oob_score=True,            # Add: provides out-of-bag validation score
            n_jobs=-1,
            random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(y, predictions):
    # Calculate metrics using sklearn (handles multi-class automatically)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, average='weighted', zero_division=0)
    recall = recall_score(y, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y, predictions, average='weighted', zero_division=0)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

def export_forest_to_json(model, filename):
    """
    Save RandomForestClassifier hyperparameters and all tree structures to a JSON file.
    Note: This is for inspection/analysis, not for reloading into sklearn.
    """
    params = model.get_params()
    trees = []
    for est in model.estimators_:
        tree = est.tree_
        tree_dict = {
            'children_left': tree.children_left.tolist(),
            'children_right': tree.children_right.tolist(),
            'feature': tree.feature.tolist(),
            'threshold': tree.threshold.tolist(),
            'value': tree.value.tolist(),
        }
        trees.append(tree_dict)
    export = {
        'hyperparameters': params,
        'n_classes': model.n_classes_,
        'classes': model.classes_.tolist(),
        'n_features': model.n_features_in_,
        'trees': trees
    }
    with open(filename, 'w') as f:
        json.dump(export, f)
    print(f"Saved forest structure and hyperparameters to {filename}")

X_train, y_train, X_test, y_test = data()
model = train_model(X_train, y_train)
joblib.dump(model, 'forestclass_model.joblib')

# Save the exact training TF-IDF vocabulary and IDF ordering from sklearn vectorizers
df_train = pd.read_csv('train.csv')
text_features, vecs = de.one_hot_encode(
    df_train,
    max_features=50,
    use_tfidf=True,
    remove_placeholders=True,
    placeholder_terms=["MODEL", "ANOTHER MODEL"],
    extra_stop_words=["this", "model"],
    return_vectorizers=True,
)

# Serialize in the same schema encoding.preprocess_row expects: ordered vocabulary and idf
import json
from sklearn.utils.validation import check_is_fitted
payload = {"columns": [], "vectorizers": [], "meta": {"max_features_per_col": 50, "min_df": 2}}
for col, vec in vecs:
    # get_feature_names_out gives the ordered feature names matching idf_
    try:
        feature_names = vec.get_feature_names_out().tolist()
    except Exception:
        # Fall back to sorting vocabulary_ by index
        inv = sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])
        feature_names = [k for k, _ in inv]
    idf = vec.idf_.tolist()
    payload["columns"].append(col)
    payload["vectorizers"].append({"column": col, "vocabulary": feature_names, "idf": idf})

with open('text_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(payload, f)
print("Saved exact training vocabulary to text_vocab.json")

y_pred = model.predict(X_test)
print("Test Set Evaluation:")
metrics = evaluate_model(y_test, y_pred)
print(metrics) 

export_forest_to_json(model, 'forest_model.json')

# Test Set Evaluation:
# {'accuracy': 0.7123287671232876, 'precision': 0.7134309557549992, 'recall': 0.7123287671232876, 'f1_score': 0.7113316536445599}

    #  64       20.0                 10                 2  0.586667   0.592157 0.586667  0.581106

# Test Set Evaluation:
# {'accuracy': 0.6986301369863014, 'precision': 0.6984929048961133, 'recall': 0.6986301369863014, 'f1_score': 0.6944560534530352}