import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    log_loss,
)
from nltk.stem import PorterStemmer

NEGATIONS = {"not", "no", "never"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-z0-9\s']", " ", text)
    # Simple tokenizer: words and contractions
    tokens = re.findall(r"[a-z0-9']+", text)
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


def load_imdb(data_dir: str = "aclImdb", sample_n: int = 50000) -> pd.DataFrame:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"IMDB data dir not found: {data_dir}")

    rows = []
    for sentiment, label in [("pos", 1), ("neg", 0)]:
        path = os.path.join(data_dir, "train", sentiment)
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".txt"):
                with open(
                    os.path.join(path, fname), "r", encoding="utf8", errors="ignore"
                ) as f:
                    text = f.read()
                rows.append({"text": clean_text(text), "label": label})

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    if sample_n is not None:
        df = df.iloc[:sample_n]
    return df


def load_sentiment140(
    data_file: str = "trainingandtestdata/training.1600000.processed.noemoticon.csv",
    sample_n: int = 40000,
) -> pd.DataFrame:
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Sentiment140 file not found: {data_file}")

    cols = ["sentiment", "id", "date", "query", "user", "text"]
    df = pd.read_csv(data_file, encoding="latin-1", header=None, names=cols)
    df["label"] = (df.sentiment == 4).astype(int)
    df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    df["text"] = df["text"].astype(str).map(clean_text)
    return df[["text", "label"]]


def plot_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
    )
    plt.title(f"Confusion Matrix ({dataset_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    out_file = f"{dataset_name.replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Confusion matrix plot saved to: {out_file}")


def plot_loss_curve(df, dataset_name):
    sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    losses = []
    pipeline = create_pipeline()

    for s in sizes:
        sample = df.sample(frac=s, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            sample["text"],
            sample["label"],
            test_size=0.2,
            stratify=sample["label"],
            random_state=42,
        )
        pipeline.fit(X_train, y_train)
        losses.append(log_loss(y_test, pipeline.predict_proba(X_test)))

    plt.figure(figsize=(8, 5))
    plt.plot([int(len(df) * s) for s in sizes], losses, marker="o")
    plt.title(f"Loss Curve ({dataset_name})")
    plt.xlabel("Train Sample Size")
    plt.ylabel("Log Loss")
    plt.grid(True)
    out_file = f"{dataset_name.replace(' ', '_')}_loss_curve.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Loss curve plot saved to: {out_file}")


def create_pipeline():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    smooth_idf=True,
                    min_df=2,
                    analyzer="word",
                ),
            ),
            ("clf", MultinomialNB(alpha=1.0)),
        ]
    )


def create_svm_pipeline():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    smooth_idf=True,
                    min_df=2,
                    analyzer="word",
                ),
            ),
            ("clf", LinearSVC(C=1.0, max_iter=10000)),
        ]
    )


def create_lr_pipeline():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    smooth_idf=True,
                    min_df=2,
                    analyzer="word",
                ),
            ),
            ("clf", LogisticRegression(max_iter=10000, C=1.0)),
        ]
    )


def make_nbsvm_model(X_train, y_train):
    vect = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        smooth_idf=True,
        min_df=2,
        analyzer="word",
    )
    X_train_vec = vect.fit_transform(X_train)
    y = np.array(y_train)

    def pr(X, y_i):
        p = X[y == y_i].sum(axis=0)
        return (p + 1) / (X[y == y_i].sum() + 1)

    r = np.log(pr(X_train_vec, 1) / pr(X_train_vec, 0))
    r = np.asarray(r).ravel()

    X_nb = X_train_vec.multiply(r)
    sv = LinearSVC(C=4.0, max_iter=10000)
    sv.fit(X_nb, y)

    return vect, r, sv


def predict_nbsvm(vect, r, model, texts):
    X = vect.transform(texts)
    return model.predict(X.multiply(r))


def compare_models(df: pd.DataFrame, dataset_name: str):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )

    model_defs = [
        ("NB", create_pipeline()),
        ("SVM", create_svm_pipeline()),
        ("Logistic", create_lr_pipeline()),
    ]
    results = []

    for name, model in model_defs:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((name, acc))

    # NBSVM
    vect, r, nbsvm_model = make_nbsvm_model(X_train, y_train)
    y_pred_nbsvm = predict_nbsvm(vect, r, nbsvm_model, X_test)
    acc_nbsvm = accuracy_score(y_test, y_pred_nbsvm)
    results.append(("NBSVM", acc_nbsvm))

    names, accs = zip(*results)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(names), y=list(accs), palette="pastel")
    plt.title(f"Model accuracy comparison ({dataset_name})")
    plt.ylim(0.6, 1.0)
    for i, v in enumerate(accs):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center")
    out_file = f"{dataset_name.replace(' ', '_')}_model_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Model comparison chart saved to: {out_file}")

    return results


def evaluate_model(df: pd.DataFrame, dataset_name: str):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["negative", "positive"], digits=4
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    loss = log_loss(y_test, pipeline.predict_proba(X_test))
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=1
    )

    print("\n==========", dataset_name, "==========")
    print("Accuracy:", acc)
    print("Log loss:", loss)
    print("5-fold CV accuracy mean:", cv_scores.mean(), "std:", cv_scores.std())
    print("\nClassification report:\n", report)
    print("\nConfusion matrix:\n", cm)

    plot_confusion_matrix(cm, dataset_name)
    plot_loss_curve(df, dataset_name)

    return pipeline


if __name__ == "__main__":
    print("Loading IMDB dataset...")
    imdb_df = load_imdb(data_dir="aclImdb", sample_n=50000)
    evaluate_model(imdb_df, "IMDB (sample 25k positive + 25k negative)")
    compare_models(imdb_df, "IMDB (sample 25k positive + 25k negative)")

    print("Loading Sentiment140 dataset...")
    twitter_df = load_sentiment140(
        data_file="trainingandtestdata/training.1600000.processed.noemoticon.csv",
        sample_n=40000,
    )
    evaluate_model(twitter_df, "Sentiment140 (40k sample)")
    compare_models(twitter_df, "Sentiment140 (40k sample)")
