import json
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import torch

from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE SETTINGS
# ============================================================
st.set_page_config(
    page_title="Hybrid Fake News Detection",
    page_icon="📰",
    layout="centered",
)

MODEL_NAME = "xlnet-base-cased"
MAX_LEN = 256
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "rf_model.joblib"
RESULTS_PATH = BASE_DIR / "hybrid_full_pipeline_results.json"
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
FAKE_DATA_PATH = DATA_DIR / "fake-real-news/Fake.csv"
TRUE_DATA_PATH = DATA_DIR / "fake-real-news/True.csv"


# ============================================================
# LOAD RESOURCES
# ============================================================
@st.cache_resource
def load_rf_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_xlnet():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


@st.cache_data
def load_kaggle_test_samples():
    fake = pd.read_csv(FAKE_DATA_PATH)
    true = pd.read_csv(TRUE_DATA_PATH)

    fake["label"] = 0
    true["label"] = 1

    kaggle = pd.concat([fake, true], ignore_index=True)

    # Same title + text construction used in the thesis pipeline
    kaggle["full_text"] = (
        (
            kaggle["title"].fillna("").astype(str)
            + " "
            + kaggle["text"].fillna("").astype(str)
        )
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Reproduce the original held-out Kaggle test split
    _, k_test = train_test_split(
        kaggle,
        test_size=0.2,
        random_state=42,
        stratify=kaggle["label"],
    )

    return k_test.reset_index(drop=True)


rf = load_rf_model()
sia = load_vader()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def clean_text(text):
    """Keep preprocessing conservative and consistent with training."""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_sentiment(text):
    scores = sia.polarity_scores(text)

    features = np.array(
        [
            [
                scores["neg"],
                scores["neu"],
                scores["pos"],
                scores["compound"],
            ]
        ],
        dtype=np.float32,
    )

    return features, scores


def get_xlnet_embedding(text):
    tokenizer, model, device = load_xlnet()

    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    last_hidden = output.last_hidden_state

    # Same mean pooling method as thesis pipeline
    mask = encoded["attention_mask"].unsqueeze(-1)

    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)

    pooled = summed / counts

    return pooled.cpu().numpy().astype(np.float32)


def build_features(text):
    semantic_features = get_xlnet_embedding(text)
    sentiment_features, sentiment_scores = get_sentiment(text)

    combined = np.hstack([semantic_features, sentiment_features])

    return combined, sentiment_scores


def predict_news(text):
    features, sentiment = build_features(text)

    if features.shape[1] != 772:
        raise ValueError(f"Expected 772 features but received {features.shape[1]}")

    prediction = int(rf.predict(features)[0])
    probabilities = rf.predict_proba(features)[0]

    class_positions = {int(label): index for index, label in enumerate(rf.classes_)}

    fake_probability = float(probabilities[class_positions[0]])

    real_probability = float(probabilities[class_positions[1]])

    return {
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "sentiment": sentiment,
    }


# ============================================================
# HEADER
# ============================================================
st.title("📰 Hybrid Fake News Detection System")

st.write("""
    **Research Prototype — MSc Information Technology, KNUST**

    This application demonstrates the hybrid fake-news detection
    framework combining **XLNet semantic embeddings**, **VADER
    sentiment analysis**, and a **Random Forest classifier**.
    """)

st.info(
    "This is a research prototype. It detects linguistic patterns "
    "associated with real and false news; it does not independently "
    "verify factual claims against external evidence."
)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Analyse News",
        "🧪 Test Samples",
        "📊 Model Performance",
        "🧠 How It Works",
    ]
)


# ============================================================
# TAB 1 — ANALYSER
# ============================================================
with tab1:

    st.subheader("Analyse a News Article")

    headline = st.text_input(
        "Headline (optional)", placeholder="Enter the news headline if available..."
    )

    article = st.text_area(
        "Article / Statement",
        height=250,
        placeholder="Paste the news article or statement here...",
    )

    combined_text = clean_text(f"{headline} {article}")

    if st.button(
        "Analyse News",
        type="primary",
        use_container_width=True,
    ):

        if len(combined_text) < 20:
            st.warning("Please enter a longer headline, article or statement.")

        else:

            with st.spinner("Generating XLNet semantic features and analysing text..."):

                try:
                    result = predict_news(combined_text)

                except Exception as exc:
                    st.error(f"Prediction could not be completed: {exc}")

                else:

                    prediction = result["prediction"]

                    if prediction == 0:
                        st.error("⚠️ Prediction: LIKELY FAKE NEWS")
                    else:
                        st.success("✅ Prediction: LIKELY REAL NEWS")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Fake Probability",
                            f"{result['fake_probability'] * 100:.1f}%",
                        )

                    with col2:
                        st.metric(
                            "Real Probability",
                            f"{result['real_probability'] * 100:.1f}%",
                        )

                    with col3:
                        confidence = max(
                            result["fake_probability"],
                            result["real_probability"],
                        )

                        st.metric("Model Confidence", f"{confidence * 100:.1f}%")

                    st.divider()

                    st.subheader("Sentiment Analysis")

                    sentiment = result["sentiment"]

                    s1, s2, s3, s4 = st.columns(4)

                    s1.metric("Negative", f"{sentiment['neg']:.3f}")

                    s2.metric("Neutral", f"{sentiment['neu']:.3f}")

                    s3.metric("Positive", f"{sentiment['pos']:.3f}")

                    s4.metric("Compound", f"{sentiment['compound']:.3f}")

                    st.caption(
                        "VADER sentiment features are combined with "
                        "768 XLNet semantic embedding features before "
                        "classification by the Random Forest."
                    )

# ============================================================
# TAB 2 — HELD-OUT TEST SAMPLES
# ============================================================
with tab2:

    st.subheader("Test the Model Using Held-Out News Samples")

    st.write("""
        This section draws examples from the **held-out Kaggle test set**
        created using the same train/test split as the research experiment.

        These samples were not included in the Kaggle training partition.
        """)

    if not FAKE_DATA_PATH.exists() or not TRUE_DATA_PATH.exists():
        st.error(
            "Fake.csv and True.csv could not be found. "
            "Place them inside the data folder."
        )

    else:
        test_samples = load_kaggle_test_samples()

        # Initialise a sample index once
        if "test_sample_index" not in st.session_state:
            st.session_state.test_sample_index = int(
                np.random.choice(test_samples.index)
            )

        col_a, col_b = st.columns([1, 1], gap="medium", vertical_alignment="bottom")

        with col_a:
            sample_type = st.selectbox(
                "Choose sample type",
                [
                    "Random",
                    "Fake only",
                    "Real only",
                ],
            )

        with col_b:
            if st.button(
                "🎲 Load New Test Sample",
                use_container_width=True,
            ):

                if sample_type == "Fake only":
                    candidates = test_samples[test_samples["label"] == 0]

                elif sample_type == "Real only":
                    candidates = test_samples[test_samples["label"] == 1]

                else:
                    candidates = test_samples

                chosen_index = int(np.random.choice(candidates.index))

                st.session_state.test_sample_index = chosen_index

        sample = test_samples.loc[st.session_state.test_sample_index]

        expected_label = int(sample["label"])

        if expected_label == 0:
            expected_text = "Fake"
        else:
            expected_text = "Real"

        st.divider()

        st.markdown("### Test Article")

        st.markdown("**Headline**")

        st.text_input(
            "Headline",
            value=str(sample.get("title", "")),
            disabled=True,
            label_visibility="collapsed",
        )

        # st.write(sample.get("title", ""))

        st.markdown("**Article**")

        # Limit display length so the interface stays clean
        article_text = str(sample.get("text", ""))

        if len(article_text) > 2500:
            display_text = article_text[:2500] + "..."
        else:
            display_text = article_text

        st.text_area(
            "Held-out article",
            value=display_text,
            height=250,
            disabled=True,
            label_visibility="collapsed",
        )

        st.markdown(f"**Known dataset label:** `{expected_text.capitalize()}`")

        if st.button(
            "▶️ Run Model on This Sample",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner("Generating XLNet features and running classification..."):

                try:
                    result = predict_news(sample["full_text"])

                except Exception as exc:
                    st.error(f"Prediction could not be completed: {exc}")

                else:
                    predicted_label = result["prediction"]

                    if predicted_label == 0:
                        predicted_text = "Fake"
                    else:
                        predicted_text = "Real"

                    correct = predicted_label == expected_label

                    st.divider()

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Expected Label",
                        expected_text.capitalize(),
                    )

                    c2.metric(
                        "Model Prediction",
                        predicted_text,
                    )

                    c3.metric(
                        "Result",
                        "Correct" if correct else "Incorrect",
                    )

                    if correct:
                        st.success("✅ The prediction matches the known label.")
                    else:
                        st.warning(
                            "⚠️ The prediction does not match the known label. "
                            "Misclassifications are expected because the model "
                            "is not perfect."
                        )

                    st.markdown("### Prediction Probabilities")

                    p1, p2 = st.columns(2)

                    p1.metric(
                        "Fake Probability", f"{result['fake_probability'] * 100:.1f}%"
                    )

                    p2.metric(
                        "Real Probability", f"{result['real_probability'] * 100:.1f}%"
                    )

                    st.markdown("### Sentiment Features")

                    sentiment = result["sentiment"]

                    s1, s2 = st.columns(2)
                    s3, s4 = st.columns(2)

                    s1.metric(
                        "Negative",
                        f"{sentiment['neg']:.3f}",
                    )

                    s2.metric(
                        "Neutral",
                        f"{sentiment['neu']:.3f}",
                    )

                    s3.metric(
                        "Positive",
                        f"{sentiment['pos']:.3f}",
                    )

                    s4.metric(
                        "Compound",
                        f"{sentiment['compound']:.3f}",
                    )

                    st.caption(
                        "The expected label comes from the original "
                        "Kaggle dataset. The prediction is generated "
                        "live using XLNet + VADER + the trained "
                        "Random Forest classifier."
                    )


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================
with tab3:

    st.subheader("Experimental Performance")

    if RESULTS_PATH.exists():

        with open(RESULTS_PATH, "r") as file:
            results = json.load(file)

        performance = pd.DataFrame(
            {
                "Dataset": [
                    "Kaggle",
                    "LIAR",
                    "BuzzFeed",
                    "PolitiFact",
                ],
                "Accuracy (%)": [
                    results["kaggle_test"]["accuracy"] * 100,
                    results["liar_holdout_test"]["accuracy"] * 100,
                    results["buzzfeed_external_test"]["accuracy"] * 100,
                    results["politifact_external_test"]["accuracy"] * 100,
                ],
            }
        )

        st.dataframe(
            performance.style.format({"Accuracy (%)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            performance,
            x="Dataset",
            y="Accuracy (%)",
        )

        st.subheader("Feature Contribution")

        importance = results["feature_importance_summary"]

        semantic = importance["semantic_percent"] * 100

        sentiment = importance["sentiment_percent"] * 100

        c1, c2 = st.columns(2)

        c1.metric("XLNet Semantic Features", f"{semantic:.2f}%")

        c2.metric("VADER Sentiment Features", f"{sentiment:.2f}%")

        st.warning(
            "Performance decreases substantially on external datasets. "
            "This demonstrates the cross-domain generalisation limitation "
            "identified in the thesis."
        )

    else:
        st.warning("Performance results file was not found.")


# ============================================================
# TAB 4 — HOW IT WORKS
# ============================================================
with tab4:

    st.subheader("Hybrid Architecture")

    st.markdown("""
        ### 1. News Input
        The user enters a news headline, article or statement.

        ### 2. XLNet Semantic Representation
        XLNet converts the text into a **768-dimensional contextual
        semantic representation**.

        ### 3. VADER Sentiment Analysis
        VADER generates four sentiment features:

        **Negative • Neutral • Positive • Compound**

        ### 4. Feature Fusion
        The features are combined:

        **768 XLNet + 4 VADER = 772 features**

        ### 5. Random Forest Classification
        The trained Random Forest uses the combined representation
        to classify the text as:

        **0 → Fake**

        **1 → Real**

        ### Important Limitation
        This system performs **classification rather than factual
        verification**. It learns linguistic and semantic patterns from
        labelled datasets and does not search external evidence to prove
        whether a statement is factually correct.
        """)


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "Hybrid Fake News Detection Research Prototype | "
    "MSc Information Technology | KNUST"
)
