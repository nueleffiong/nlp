# -*- coding: utf-8 -*-
"""
fake_news.py

Converted from fake_news.ipynb (Colab).
This script trains a TF-IDF + Logistic Regression classifier to detect
fake vs real news using two CSV files (True.csv and Fake.csv).

Usage:
    python fake_news.py --true True.csv --fake Fake.csv
    python fake_news.py --true path/to/True.csv --fake path/to/Fake.csv --use-bert

Notes:
- This script is intended to run outside Colab. It does not perform pip installs.
  If a required package is missing, install it first, e.g.:
    pip install pandas scikit-learn matplotlib transformers
"""

from __future__ import annotations
import os
import sys
import argparse
import textwrap
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Required imports (fail fast with helpful message)
try:
    import pandas as pd
except Exception as e:
    print("Missing dependency: pandas. Install with `pip install pandas`.")
    raise e

try:
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except Exception as e:
    print("Missing dependency: scikit-learn. Install with `pip install scikit-learn`.")
    raise e

# Optional imports
try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Transformers is optional and only used if --use-bert is passed
try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

def load_and_prepare(true_path: str, fake_path: str) -> pd.DataFrame:
    """Load True.csv and Fake.csv, add labels, combine and clean."""
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True file not found: {true_path}")
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake file not found: {fake_path}")

    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    # Add label columns
    true_df = true_df.copy()
    fake_df = fake_df.copy()
    true_df["label"] = "real"
    fake_df["label"] = "fake"

    # Combine
    data = pd.concat([true_df, fake_df], ignore_index=True)

    # Basic checks and cleaning
    if "text" not in data.columns:
        raise KeyError("Expected a 'text' column in both CSV files.")

    data = data.dropna(subset=["text", "label"]).reset_index(drop=True)
    data["label_num"] = data["label"].map({"real": 1, "fake": 0})

    return data

def train_tf_logreg(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train TF-IDF vectorizer and Logistic Regression classifier."""
    X = data["text"]
    y = data["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["fake", "real"])
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "vectorizer": vectorizer,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
    }
    return results

def print_summary(data: pd.DataFrame, results: dict):
    print("\nData summary:")
    print(f" - total examples: {len(data)}")
    print(" - label counts:")
    print(data["label"].value_counts().to_string())

    print("\nModel evaluation:")
    print(f" - Accuracy: {results['accuracy']:.4f}")
    print("\nClassification report:")
    print(results["report"])
    print("Confusion matrix:")
    print(results["confusion_matrix"])  

def interactive_predict(model, vectorizer):
    """Prompt user for a headline/text and print prediction."""
    try:
        while True:
            text = input("\nEnter a headline/text to classify (or press Enter to quit): ").strip()
            if not text:
                print("Exiting interactive prediction.")
                break
            tfidf = vectorizer.transform([text])
            pred = model.predict(tfidf)[0]
            label = "real" if int(pred) == 1 else "fake"
            print(f"Prediction: {label} (label_num={{int(pred)}})")
    except (KeyboardInterrupt, EOFError):
        print("\nInteractive prediction terminated.")

def plot_top_features(vectorizer, model, top_n: int = 10, out_path: str | None = None):
    """Plot top positive and negative features from logistic regression coefficients."""
    if np is None:
        print("NumPy not available; cannot compute top features.")
        return
    if plt is None:
        print("matplotlib not available; cannot plot top features.")
        return

    try:
        feat_names = np.array(vectorizer.get_feature_names_out())
        coefs = model.coef_[0]
        # top positive
        top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
        top_neg_idx = np.argsort(coefs)[:top_n]

        labels = np.concatenate([feat_names[top_pos_idx], feat_names[top_neg_idx]])
        values = np.concatenate([coefs[top_pos_idx], coefs[top_neg_idx]])

        y_pos = np.arange(len(labels))

        plt.figure(figsize=(10, 6))
        colors = ["tab:green"] * top_n + ["tab:red"] * top_n
        plt.barh(y_pos, values, color=colors)
        plt.yticks(y_pos, labels)
        plt.xlabel("Coefficient value (positive -> indicates 'real')")
        plt.title(f"Top {top_n} positive and top {top_n} negative features")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if out_path:
            plt.savefig(out_path)
            print(f"Saved feature plot to {out_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"Could not generate feature plot: {e}")

def run_transformer_demo(sample_text: str = "Breaking news: Trump is our president"):
    """Run a small transformer-based text-classification demo if transformers installed."""
    if hf_pipeline is None:
        print("transformers not installed; skipping transformer demo. Install with `pip install transformers`.")
        return
    try:
        # Using a lightweight pretrained model for demo
        clf = hf_pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = clf(sample_text)[0]
        print("\nTransformer demo result:")
        print(f" - text: {sample_text} ")
        print(f" - label: {result.get('label')}, score: {result.get('score'):.4f}")
    except Exception as e:
        print(f"Transformer demo failed: {e}")

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a TF-IDF + Logistic Regression model to detect fake news.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """Example:
  python fake_news.py --true True.csv --fake Fake.csv --plot-features features.png --use-bert
"""
        ),
    )
    p.add_argument("--true", "-t", default="True.csv", help="Path to True.csv")
    p.add_argument("--fake", "-f", default="Fake.csv", help="Path to Fake.csv")
    p.add_argument("--no-interactive", dest="interactive", action="store_false", help="Skip interactive prediction")
    p.add_argument("--use-bert", action="store_true", help="Run a small transformer demo (requires `transformers`)" )
    p.add_argument("--plot-features", nargs="?", const="top_features.png", help="Save top feature plot to file (requires matplotlib & numpy).")
    p.add_argument("--test-size", type=float, default=0.2, help="Test set proportion (default: 0.2)")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    try:
        data = load_and_prepare(args.true, args.fake)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    results = train_tf_logreg(data, test_size=args.test_size)
    print_summary(data, results)

    # Optionally plot top features
    if args.plot_features:
        plot_top_features(results["vectorizer"], results["model"], top_n=10, out_path=args.plot_features)

    # Optional transformer demo
    if args.use_bert:
        run_transformer_demo()

    # Interactive prediction (unless disabled)
    if args.interactive:
        interactive_predict(results["model"], results["vectorizer"])  

if __name__ == "__main__":
    main()