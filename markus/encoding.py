
import sys
import csv
import numpy as np
import pandas as pd

# Column names
BEST_TASKS_COL = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
SUBOPT_TASKS_COL = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
ACADEMIC_COL = "How likely are you to use this model for academic tasks?"
SUBOPT_FREQ_COL = "Based on your experience, how often has this model given you a response that felt suboptimal?"
TEXT_COLS = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
]

TARGET_TASKS = [
    'Math computations',
    'Writing or debugging code',
    'Data processing or analysis',
    'Explaining complex concepts simply',
]

PLACEHOLDER_TERMS = ["MODEL", "ANOTHER MODEL"]
EXTRA_STOP_WORDS = ["this", "model"]
MIN_TOKEN_LEN = 3
MAX_FEATURES_PER_COL = 50
MIN_DF = 2  # minimum document frequency

# Built-in English stop words (complete sklearn list)
ENGLISH_STOP_WORDS = frozenset([
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost',
    'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst',
    'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere',
    'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming',
    'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between',
    'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con',
    'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during',
    'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc',
    'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen',
    'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found',
    'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have',
    'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed',
    'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least',
    'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more',
    'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely',
    'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor',
    'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto',
    'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part',
    'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming',
    'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six',
    'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere',
    'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves',
    'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these',
    'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout',
    'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two',
    'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what',
    'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein',
    'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole',
    'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your',
    'yours', 'yourself', 'yourselves'
])

ALL_STOP_WORDS = ENGLISH_STOP_WORDS | set(EXTRA_STOP_WORDS)

# ============================================================================
# Helper functions
# ============================================================================

def extract_rating(response):
    """Extract numeric rating from responses like '3 - Sometimes'."""
    if response is None or (isinstance(response, float) and np.isnan(response)):
        return None
    s = str(response).strip()
    num = ''
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            break
    return int(num) if num else None

def remove_placeholders(text):
    """Remove bracketed spans containing any placeholder term (case-insensitive)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ''
    s = str(text)
    result = ''
    i = 0
    while i < len(s):
        if s[i] == '[':
            j = s.find(']', i + 1)
            if j == -1:
                result += s[i:]
                break
            span = s[i+1:j]
            span_upper = span.upper()
            if any(term.upper() in span_upper for term in PLACEHOLDER_TERMS):
                i = j + 1
                continue
            else:
                result += ' ' + span + ' '
                i = j + 1
                continue
        else:
            result += s[i]
            i += 1
    return ' '.join(result.split())

def tokenize(text):
    """Tokenize using sklearn-like token_pattern: r"\b[a-zA-Z]{3,}\b".    
    Apply stop word filtering to match sklearn's behavior.
    """
    cleaned = remove_placeholders(text).lower()
    # Manual regex replacement: extract 3+ letter alphabetic tokens
    tokens = []
    current_token = ''
    for char in cleaned:
        if char.isalpha():
            current_token += char
        else:
            if len(current_token) >= MIN_TOKEN_LEN:
                tokens.append(current_token)
            current_token = ''
    # Don't forget the last token
    if len(current_token) >= MIN_TOKEN_LEN:
        tokens.append(current_token)
    # Filter stop words (matching sklearn's behavior)
    return [tok for tok in tokens if tok not in ALL_STOP_WORDS]

def process_multiselect(series, target_tasks):
    """Convert multiselect strings to lists, keeping only specified features."""
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed

def manual_multilabel_binarizer(lists, all_classes):
    """Manual MultiLabelBinarizer: convert list of lists to binary matrix."""
    n_samples = len(lists)
    n_classes = len(all_classes)
    result = np.zeros((n_samples, n_classes), dtype=int)
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    for i, item_list in enumerate(lists):
        for item in item_list:
            if item in class_to_idx:
                result[i, class_to_idx[item]] = 1
    return result

# ============================================================================
# TF-IDF vectorizer (manual implementation)
# ============================================================================

class ManualTfidfVectorizer:
    """Manual TF-IDF vectorizer matching sklearn behavior (max_features, min_df)."""
    
    def __init__(self, max_features=50, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_ = []
        self.idf_ = []
        
    def fit(self, texts):
        """Build vocabulary from texts."""
        # Tokenize all documents
        tokenized_docs = [tokenize(text) for text in texts]
        
        # Count document frequency for each token (manual Counter replacement)
        df_counter = {}
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df_counter[token] = df_counter.get(token, 0) + 1
        
        # Filter by min_df
        valid_tokens = {token for token, count in df_counter.items() if count >= self.min_df}
        
        # Select top max_features by total frequency (match sklearn ordering: by freq desc, then alpha)
        token_freq = {}
        for tokens in tokenized_docs:
            for token in tokens:
                if token in valid_tokens:
                    token_freq[token] = token_freq.get(token, 0) + 1

        # Sort by (-freq, token) and keep top max_features
        ranked = sorted(token_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        top_tokens = ranked[: self.max_features]
        # Preserve this order in the vocabulary
        self.vocabulary_ = [tok for tok, _ in top_tokens]
        
        # Compute IDF for each token in vocabulary
        n_docs = len(texts)
        self.idf_ = []
        for token in self.vocabulary_:
            df = df_counter[token]
            # IDF formula: log((n_docs + 1) / (df + 1)) + 1 (sklearn's smooth_idf=True default)
            idf = np.log((n_docs + 1) / (df + 1)) + 1
            self.idf_.append(idf)
        
        return self
    
    def transform(self, texts):
        """Transform texts to TF-IDF matrix."""
        n_docs = len(texts)
        n_features = len(self.vocabulary_)
        matrix = np.zeros((n_docs, n_features), dtype=float)
        
        vocab_index = {token: idx for idx, token in enumerate(self.vocabulary_)}
        
        for doc_idx, text in enumerate(texts):
            tokens = tokenize(text)
            # Count term frequency (manual Counter replacement)
            tf_counter = {}
            for token in tokens:
                tf_counter[token] = tf_counter.get(token, 0) + 1
            # Apply TF-IDF
            for token, tf in tf_counter.items():
                if token in vocab_index:
                    idx = vocab_index[token]
                    # TF-IDF = TF * IDF (sklearn uses raw count as TF by default for TfidfVectorizer)
                    matrix[doc_idx, idx] = tf * self.idf_[idx]
        
        # L2 normalization (row-wise)
        for i in range(n_docs):
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
        
        return matrix
    
    def fit_transform(self, texts):
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

# ============================================================================
# Main preprocess function
# ============================================================================

def preprocess(df, return_vectorizers: bool = False):
    """
    Main preprocessing function matching data_encoding.preprocess() exactly.
    
    Parameters:
        df: pandas DataFrame with survey data
        
    Returns:
        X: numpy array of features (shape: n_samples x n_features)
        y: numpy array of labels (strings: 'ChatGPT', 'Claude', 'Gemini')
    """
    # Drop rows with missing data
    df = df.dropna().copy()
    
    # Extract and normalize rating features
    academic_numeric = df[ACADEMIC_COL].apply(extract_rating)
    subopt_numeric = df[SUBOPT_FREQ_COL].apply(extract_rating)
    
    # Fill NaN with median
    academic_numeric = academic_numeric.fillna(academic_numeric.median())
    subopt_numeric = subopt_numeric.fillna(subopt_numeric.median())
    
    # Normalize to 0-1
    academic_norm = (academic_numeric - 1) / 4.0
    subopt_norm = (subopt_numeric - 1) / 4.0
    
    # Process multi-select columns
    best_tasks_lists = process_multiselect(df[BEST_TASKS_COL], TARGET_TASKS)
    suboptimal_tasks_lists = process_multiselect(df[SUBOPT_TASKS_COL], TARGET_TASKS)
    
    # Manual multilabel binarization
    best_tasks_encoded = manual_multilabel_binarizer(best_tasks_lists, TARGET_TASKS)
    suboptimal_tasks_encoded = manual_multilabel_binarizer(suboptimal_tasks_lists, TARGET_TASKS)
    
    # Process text columns with TF-IDF
    text_features = []
    fitted_vectorizers = []
    for col in TEXT_COLS:
        if col in df.columns:
            texts = df[col].fillna('')
            vectorizer = ManualTfidfVectorizer(max_features=MAX_FEATURES_PER_COL, min_df=MIN_DF)
            features = vectorizer.fit_transform(texts)
            text_features.append(features)
            fitted_vectorizers.append((col, vectorizer))
    
    # Combine all features
    feature_parts = [
        academic_norm.values.reshape(-1, 1),
        subopt_norm.values.reshape(-1, 1),
        best_tasks_encoded,
        suboptimal_tasks_encoded
    ] + text_features
    
    X = np.hstack(feature_parts)
    
    # Extract labels
    # y = df['label'].values
    
    if return_vectorizers:
        return X, fitted_vectorizers
    return X

def save_text_vocab(df: pd.DataFrame, output_path: str = 'text_vocab.json'):
    """Fit text vectorizers on df and save vocabulary + idf arrays to JSON for later single-row inference.

    JSON schema:
    {
        "columns": [col1, col2, col3],
        "vectorizers": [
            {"column": col, "vocabulary": [...], "idf": [...]} , ...
        ],
        "meta": {"max_features_per_col": 50, "min_df": 2}
    }
    """
    import json
    _, _, vecs = preprocess(df, return_vectorizers=True)
    payload = {
        "columns": [c for c, _ in vecs],
        "vectorizers": [
            {"column": c, "vocabulary": v.vocabulary_, "idf": v.idf_} for c, v in vecs
        ],
        "meta": {"max_features_per_col": MAX_FEATURES_PER_COL, "min_df": MIN_DF}
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    return output_path

def load_text_vocab(path: str):
    """Load previously saved vocab JSON."""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_text_row(row: pd.Series, vocab_json: dict):
    """Produce concatenated TF-IDF row vector using stored vocabulary & idf.
    """
    parts = []
    for vec in vocab_json["vectorizers"]:
        col = vec["column"]
        vocab = vec["vocabulary"]
        idf = vec["idf"]
        text = row.get(col, '')
        tokens = tokenize(text)
        # Count term frequency (manual Counter replacement)
        tf_counter = {}
        for token in tokens:
            tf_counter[token] = tf_counter.get(token, 0) + 1
        arr = np.zeros(len(vocab), dtype=float)
        index = {tok: i for i, tok in enumerate(vocab)}
        for token, tf in tf_counter.items():
            idx = index.get(token)
            if idx is not None:
                arr[idx] = tf * idf[idx]
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        parts.append(arr)
    if parts:
        return np.concatenate(parts)
    return np.array([], dtype=float)

def preprocess_row(row: pd.Series, vocab_json: dict):
    """Build full feature vector for a single row using saved vocab (no refit).
    """
    # Ratings
    academic_numeric = extract_rating(row.get(ACADEMIC_COL)) or 3
    subopt_numeric = extract_rating(row.get(SUBOPT_FREQ_COL)) or 3
    academic_norm = (academic_numeric - 1) / 4.0
    subopt_norm = (subopt_numeric - 1) / 4.0

    # Multi-select
    best_list = process_multiselect(pd.Series([row.get(BEST_TASKS_COL)]), TARGET_TASKS)[0]
    subopt_list = process_multiselect(pd.Series([row.get(SUBOPT_TASKS_COL)]), TARGET_TASKS)[0]
    best_vec = manual_multilabel_binarizer([best_list], TARGET_TASKS)[0]
    subopt_vec = manual_multilabel_binarizer([subopt_list], TARGET_TASKS)[0]

    text_vec = encode_text_row(row, vocab_json)

    return np.concatenate([
        np.array([academic_norm, subopt_norm]),
        best_vec,
        subopt_vec,
        text_vec
    ])
