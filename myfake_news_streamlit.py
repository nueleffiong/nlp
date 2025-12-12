
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle


# ----------------- Page config -----------------
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("Fake News Detector — TF-IDF + Logistic Regression")
st.markdown(
    "Upload the True.csv and Fake.csv files, train the model, "
    "view evaluation metrics, and predict whether input text is real or fake."
)

# ----------------- Session state -----------------
if "model_info" not in st.session_state:
    st.session_state.model_info = None

if "data" not in st.session_state:
    st.session_state.data = None


# ----------------- Sidebar controls -----------------
st.sidebar.header("Data / Training settings")

use_uploaded = st.sidebar.checkbox("Upload CSV files", value=True)
true_upload = None
fake_upload = None

if use_uploaded:
    true_upload = st.sidebar.file_uploader("Upload True.csv", type=["csv"])
    fake_upload = st.sidebar.file_uploader("Upload Fake.csv", type=["csv"])
else:
    st.sidebar.write("Provide file paths on the server (not supported in this demo).")

test_size = st.sidebar.slider("Test set fraction", 0.05, 0.5, 0.2)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)
train_button = st.sidebar.button("Load & Train")

st.sidebar.markdown("---")
st.sidebar.markdown("Optional")

run_transformers = st.sidebar.checkbox(
    "Run lightweight transformer demo (requires transformers)", value=False
)

plot_output_name = st.sidebar.text_input(
    "Save feature plot as (leave blank to show only)", value=""
)

info_placeholder = st.empty()


# ----------------- Helpers -----------------
@st.cache_data
def load_csv_file(uploaded: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame | None:
    """Read an uploaded CSV into a DataFrame."""
    if uploaded is None:
        return None
    try:
        df = pd.read_csv(uploaded)
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None


@st.cache_resource
def train_model_and_vectorizer(df: pd.DataFrame, test_size: float, random_state: int) -> dict:
    """Train TF‑IDF + Logistic Regression model and return artifacts."""
    if "text" not in df.columns:
        raise KeyError("Combined data must contain a 'text' column.")

    X = df["text"].astype(str)
    y = df["label_num"].astype(int)

    X, y = shuffle(X, y, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["fake", "real"],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    return {
        "vectorizer": vectorizer,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "feature_names": vectorizer.get_feature_names_out(),
        "coefs": model.coef_[0],
    }


def prepare_combined(true_df: pd.DataFrame, fake_df: pd.DataFrame) -> pd.DataFrame:
    """Combine True/Fake data, ensure text + numeric labels."""
    true_df = true_df.copy()
    fake_df = fake_df.copy()

    true_df["label"] = "real"
    fake_df["label"] = "fake"

    data = pd.concat([true_df, fake_df], ignore_index=True)

    if "text" not in data.columns:
        if "title" in data.columns:
            data["text"] = data["title"].astype(str)
        else:
            raise KeyError("Neither 'text' nor 'title' columns found in uploaded CSVs.")

    data = data.dropna(subset=["text", "label"]).reset_index(drop=True)
    data["label_num"] = data["label"].map({"real": 1, "fake": 0})

    return data


# ----------------- Load & Train flow -----------------
if train_button:
    if not use_uploaded:
        st.sidebar.error("Non-upload file-loading is not available in this demo.")
    else:
        if true_upload is None or fake_upload is None:
            st.sidebar.error("Please upload both True.csv and Fake.csv to train.")
        else:
            with st.spinner("Loading CSVs..."):
                true_df = load_csv_file(true_upload)
                fake_df = load_csv_file(fake_upload)

            if true_df is None or fake_df is None:
                st.error("Failed to load one or more uploaded CSVs.")
                st.session_state.data = None
                st.session_state.model_info = None
            else:
                try:
                    data = prepare_combined(true_df, fake_df)
                    st.session_state.data = data
                except Exception as e:
                    st.error(f"Error preparing data: {e}")
                    st.session_state.data = None
                    st.session_state.model_info = None

            if st.session_state.data is not None:
                info_placeholder.info(
                    f"Loaded dataset with {len(st.session_state.data)} rows — training..."
                )
                try:
                    model_info = train_model_and_vectorizer(
                        st.session_state.data,
                        test_size=float(test_size),
                        random_state=int(random_state),
                    )
                    st.session_state.model_info = model_info
                    info_placeholder.success(
                        f"Training complete — accuracy: {model_info['accuracy']:.4f}"
                    )
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.session_state.model_info = None

data = st.session_state.data
model_info = st.session_state.model_info

if use_uploaded and (true_upload is not None and fake_upload is not None) and model_info is None:
    st.info("Files uploaded. Press 'Load & Train' in the sidebar to train the model.")

# ----------------- Data sample -----------------
if data is not None:
    st.subheader("Data sample")
    st.dataframe(data.head())

# ----------------- Evaluation & explainability -----------------
if model_info is not None:
    st.subheader("Model evaluation")
    st.write(f"Accuracy: **{model_info['accuracy']:.4f}**")
    st.text("Classification report:")
    st.text(model_info["report"])

    # Confusion matrix
    fig_cm, ax = plt.subplots(figsize=(4, 3))
    cm = model_info["confusion_matrix"]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["fake", "real"])
    ax.set_yticklabels(["fake", "real"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    st.pyplot(fig_cm)

    # Top features
    st.subheader("Top features (from Logistic Regression coefficients)")
    try:
        feature_names = np.array(model_info["feature_names"])
        coefs = model_info["coefs"]
        top_n = 10

        top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
        top_neg_idx = np.argsort(coefs)[:top_n]

        labels = np.concatenate(
            [feature_names[top_pos_idx], feature_names[top_neg_idx]]
        )
        values = np.concatenate([coefs[top_pos_idx], coefs[top_neg_idx]])
        colors = ["tab:green"] * top_n + ["tab:red"] * top_n

        fig_feats, ax2 = plt.subplots(figsize=(8, 6))
        y_pos = np.arange(len(labels))
        ax2.barh(y_pos, values, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yt
