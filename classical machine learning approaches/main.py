# ============================ Imports ============================ #
import os
import re
import string
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from nltk.util import ngrams
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.metrics import mean_squared_error as root_mean_squared_error
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

# ======================= Process Essays ========================= #
def process_essays(
    essays,
    model_name=None,
    model_type=None,
    pooling_strategy="avg",
    stopwords_path=None,
    cache_dir=None,
    features=False,
    embeddings=False,
    qwk=False,
    y_true=None,
    y_pred=None
):
    """Process essays: optional features, embeddings, QWK scoring."""

    def _load_stopwords(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(word.strip() for word in f if word.strip())

    def _arabic_sentence_splitter(text):
        return [s for s in re.split(r"[.!؟]\s*", text.strip()) if s]

    def _tokenize(text):
        return re.findall(r"[\u0600-\u06FFa-zA-Z0-9]+", text)

    def _analyze_single_essay(text, stopwords):
        arabic_punctuations = "؟،؛«»…“”ـ"
        if not isinstance(text, str) or not text.strip():
            return np.zeros(15, dtype=np.float32)
        characters = len(text)
        words = _tokenize(text)
        unique_words = set(words)
        sentences = _arabic_sentence_splitter(text)
        puncts = [c for c in text if c in string.punctuation or c in arabic_punctuations]
        stopword_count = sum(1 for w in words if w in stopwords)
        sentence_lengths = [len(_tokenize(s)) for s in sentences]
        total_words = len(words)
        total_unique_words = len(unique_words)
        total_sentences = len(sentences)
        bigrams = list(ngrams(words, 2))
        trigrams = list(ngrams(words, 3))
        total_bigrams = len(bigrams)
        total_trigrams = len(trigrams)
        unique_bigrams = len(set(bigrams))
        unique_trigrams = len(set(trigrams))
        bigram_diversity = unique_bigrams / total_bigrams if total_bigrams else 0
        trigram_diversity = unique_trigrams / total_trigrams if total_trigrams else 0
        features_dict = {
            "avg_word_count": total_words,
            "avg_unique_word_count": total_unique_words,
            "avg_punctuation_count": len(puncts),
            "avg_sentence_count": total_sentences,
            "avg_word_length": sum(len(w) for w in words) / total_words if total_words else 0,
            "avg_sentence_length_in_words": sum(sentence_lengths) / total_sentences if total_sentences else 0,
            "avg_char_count": characters,
            "avg_stopword_count": stopword_count,
            "bigram_count": total_bigrams,
            "trigram_count": total_trigrams,
            "unique_bigram_count": unique_bigrams,
            "unique_trigram_count": unique_trigrams,
            "bigram_diversity": bigram_diversity,
            "trigram_diversity": trigram_diversity,
        }
        return np.array(list(features_dict.values()), dtype=np.float32)

    def _compute_embeddings(essays_list, model_name, model_type, pooling, cache_dir):
        if cache_dir is None:
            cache_dir = "./embeddings_cache"
        os.makedirs(cache_dir, exist_ok=True)
        emb_file = os.path.join(cache_dir, f"{model_name.replace('/', '_')}_{len(essays_list)}essay_embeddings.npy")
        if os.path.exists(emb_file):
            return np.load(emb_file)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        embeddings = []
        with torch.no_grad():
            for text in tqdm(essays_list, desc="Embedding"):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                if pooling == "avg":
                    masked_hidden = last_hidden_state * mask
                    summed = masked_hidden.sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    pooled = summed / counts
                elif pooling == "cls":
                    pooled = last_hidden_state[:, 0, :]
                elif pooling == "max":
                    masked_hidden = last_hidden_state.masked_fill(mask == 0, float("-inf"))
                    pooled = masked_hidden.max(dim=1).values
                elif pooling == "avg+cls":
                    avg_pool = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    cls_pool = last_hidden_state[:, 0, :]
                    pooled = torch.cat((cls_pool, avg_pool), dim=1)
                elif pooling == "avg+max+cls":
                    avg_pool = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    max_pool = last_hidden_state.masked_fill(mask == 0, float("-inf")).max(dim=1).values
                    cls_pool = last_hidden_state[:, 0, :]
                    pooled = torch.cat((cls_pool, avg_pool, max_pool), dim=1)
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling}")
                embeddings.append(pooled.squeeze().numpy())
        embeddings = np.array(embeddings)
        np.save(emb_file, embeddings)
        return embeddings

    def _compute_qwk(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
        qwk_scores = []
        for i in range(y_true.shape[1]):
            y_true_col = y_true[:, i].round().astype(int)
            y_pred_col = np.clip(y_pred[:, i], 0, 5).round().astype(int)
            score = cohen_kappa_score(y_true_col, y_pred_col, weights="quadratic")
            qwk_scores.append(score)
        return np.mean(qwk_scores)

    results = {}
    if features:
        stopwords = _load_stopwords(stopwords_path)
        results["features"] = np.array([_analyze_single_essay(e, stopwords) for e in essays])
    if embeddings:
        results["embeddings"] = _compute_embeddings(essays, model_name, model_type, pooling_strategy, cache_dir)
    if qwk:
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred are required for QWK computation.")
        results["qwk"] = _compute_qwk(y_true, y_pred)
    return results

# ---------------- Scorer ---------------- #
def qwk_scorer_func(y_true, y_pred):
    return process_essays(None, qwk=True, y_true=y_true, y_pred=y_pred)["qwk"]

qwk_sklearn_scorer = make_scorer(
    qwk_scorer_func,
    greater_is_better=True
)

# ================== Model Pipeline ================== #
def build_model_pipeline(
    model_name,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring=None
):
    if scoring is None:
        scoring = {"QWK": qwk_sklearn_scorer}

    configs = {
        "ridge": {"estimator": Ridge(), "params": {"regressor__alpha": [0.0001,0.001,0.01,0.1,1,10,100,1000], "regressor__solver": ["auto","svd","cholesky","lsqr","sag","saga"]}},
        "elasticnet": {"estimator": ElasticNet(max_iter=5000), "params":{"regressor__alpha":[0.0001,0.001,0.01,0.1,1,10,100], "regressor__l1_ratio":[0,0.1,0.25,0.5,0.75,0.9,1.0], "regressor__selection":["cyclic","random"]}},
        "lasso":{"estimator":Lasso(max_iter=10000),"params":{"regressor__alpha":[0.0001,0.001,0.01,0.1,1,10,100],"regressor__selection":["cyclic","random"]}},
        "rf":{"estimator":MultiOutputRegressor(RandomForestRegressor(random_state=42)), "params":{"regressor__estimator__n_estimators":[100,300],"regressor__estimator__max_depth":[10,30],"regressor__estimator__min_samples_split":[2,5],"regressor__estimator__min_samples_leaf":[1,2],"regressor__estimator__max_features":["sqrt",None],"regressor__estimator__bootstrap":[True]}},
        "xgb":{"estimator":MultiOutputRegressor(XGBRegressor(tree_method="hist", random_state=42)), "params":{"regressor__estimator__n_estimators":[200],"regressor__estimator__max_depth":[5,10],"regressor__estimator__learning_rate":[0.01,0.1],"regressor__estimator__subsample":[0.8],"regressor__estimator__colsample_bytree":[0.8],"regressor__estimator__reg_alpha":[0,0.5],"regressor__estimator__reg_lambda":[1.0],"regressor__estimator__gamma":[0]}}
    }

    if model_name not in configs:
        raise ValueError(f"Unsupported model: {model_name}")

    pipeline = Pipeline([("regressor", configs[model_name]["estimator"])])
    grid = GridSearchCV(estimator=pipeline, param_grid=configs[model_name]["params"], cv=cv, scoring=scoring, refit="QWK", n_jobs=n_jobs, verbose=verbose)
    return grid

# ================== Train & Evaluate ================== #
def train_evaluate_final_model(
    tag, train_data, test_data, head_model, embedding_model, embedding_model_type, save_path, score_columns,
    include_features=False, pooling_strategy="avg", stopwords_path=None, cache_dir=None
):
    os.makedirs(save_path, exist_ok=True)
    safe_model_name = embedding_model.replace("/", "_")

    def _prepare_embeddings(df):
        embs = process_essays(df["essay"].tolist(), model_name=embedding_model, model_type=embedding_model_type, pooling_strategy=pooling_strategy, stopwords_path=stopwords_path, cache_dir=cache_dir, features=False, embeddings=True)["embeddings"]
        if include_features:
            # Pass stopwords_path to process_essays when computing features
            feats = process_essays(df["essay"].tolist(), features=True, stopwords_path=stopwords_path)["features"]
            embs = np.hstack((embs, feats))
        return embs

    embeddings_train = _prepare_embeddings(train_data)
    embeddings_test = _prepare_embeddings(test_data)
    train_scores = train_data[score_columns]
    test_scores = test_data[score_columns]

    model = build_model_pipeline(head_model)
    model.fit(embeddings_train, train_scores)
    preds_test = model.predict(embeddings_test)

    # Save predictions
    pd.DataFrame(preds_test, columns=score_columns).to_csv(os.path.join(save_path, f"{tag}_{safe_model_name}_{head_model}_predictions.csv"), index=False)
    joblib.dump(model, os.path.join(save_path, f"{tag}_{safe_model_name}_{head_model}_final_model.joblib"))

    results = {}
    for i, col in enumerate(score_columns):
        test_qwk = cohen_kappa_score(test_scores[col].round().astype(int), np.clip(preds_test[:, i], 0, 5).round().astype(int), weights="quadratic")
        test_rmse = root_mean_squared_error(test_scores[col], preds_test[:, i])
        results[f"{col} Test RMSE"] = test_rmse
        results[f"{col} Test QWK"] = test_qwk
        print(f"[{col}] Test RMSE: {test_rmse:.4f}, QWK: {test_qwk:.4f}")
    pd.DataFrame([results]).to_csv(os.path.join(save_path, f"{tag}_{safe_model_name}_{head_model}_evaluation_results.csv"), index=False)

    if hasattr(model, "best_params_"):
        with open(os.path.join(save_path, f"{tag}_{safe_model_name}_{head_model}_best_parameters.json"), "w") as f:
            json.dump(model.best_params_, f, indent=4)

# ================== Run Experiment ================== #
def run_experiment(tag, train_data, test_data, head_model, embedding_model, embedding_model_type, save_path, score_columns, include_features=False, pooling_strategy="avg", stopwords_path=None, cache_dir=None):
    print("="*80)
    print(f"🚀 Starting experiment: {tag} | Model: {head_model} | Embeddings: {embedding_model}")
    print("="*80)
    train_evaluate_final_model(tag, train_data, test_data, head_model, embedding_model, embedding_model_type, save_path, score_columns, include_features, pooling_strategy, stopwords_path, cache_dir)
    print("="*80)
    print(f"✅ Experiment {tag} completed.")
    print("="*80)

# ================= Example Usage ================= #
x = pd.read_json("/content/TAQEEM2025_TaskB_train_essays.json")
y = pd.read_csv("/content/TAQEEM2025_TaskB_train_human_scores.csv")
merged_df = pd.merge(x, y, on=['prompt_id', 'essay_id'])
train = merged_df[merged_df["prompt_id"] == 1]
test = merged_df[merged_df["prompt_id"] == 2]

# score_cols = ["relevance", "organization", "vocabulary", "style", "development", "mechanics", "grammar"]
# run_experiment("exp1", train, test, "lasso", "CAMeL-Lab/bert-base-arabic-camelbert-mix", "hf", "./results", score_cols, include_features=True, stopwords_path="/content/arabic_stopwords.txt")
