# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# --- Page setup ---
st.set_page_config(
    page_title="🌸 Iris Species Classifier",
    page_icon="🌼",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for style ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f9fbfd;
        color: #222;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        color: #4a148c;
        text-align: center;
        font-size: 2.2rem;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #666;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model ---
@st.cache_data
def load_model():
    model_data = joblib.load("model.joblib")
    return model_data

try:
    data = load_model()
    model = data["model"]
    feature_names = data["feature_names"]
    target_names = data["target_names"]
except Exception as e:
    st.error(f"⚠️ Could not load model.joblib: {e}")
    st.stop()

# --- Sidebar Mode Toggle ---
st.sidebar.title("🔧 Navigation")
mode = st.sidebar.radio("Select Mode", ["Predict 🌿", "Explore 📊"])

# --- Title & subtitle ---
st.markdown("<h1 class='main-title'>🌼 Iris Flower Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by scikit-learn · Built with Streamlit · by Pranav Datta Y Sharma</p>", unsafe_allow_html=True)

# --- Predict Mode ---
if "Predict" in mode:
    st.subheader("🔮 Make a Prediction")

    col1, col2 = st.columns(2)
    mins = {'sepal length (cm)':4.3, 'sepal width (cm)':2.0, 'petal length (cm)':1.0, 'petal width (cm)':0.1}
    maxs = {'sepal length (cm)':7.9, 'sepal width (cm)':4.4, 'petal length (cm)':6.9, 'petal width (cm)':2.5}
    defaults = {'sepal length (cm)':5.8, 'sepal width (cm)':3.0, 'petal length (cm)':3.7, 'petal width (cm)':1.0}

    inputs = []
    for i, fname in enumerate(feature_names):
        if i % 2 == 0:
            val = col1.slider(fname, mins[fname], maxs[fname], defaults[fname], step=0.1)
        else:
            val = col2.slider(fname, mins[fname], maxs[fname], defaults[fname], step=0.1)
        inputs.append(val)

    if st.button("🌿 Predict"):
        X = np.array(inputs).reshape(1, -1)
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        st.success(f"✅ Predicted Species: **{target_names[pred].title()}**")

        prob_df = pd.DataFrame({
            "Species": target_names,
            "Probability": [float(p) for p in probs]
        })
        st.bar_chart(prob_df.set_index("Species"))

# --- Explore Mode ---
else:
    st.subheader("📊 Explore the Iris Dataset")
    iris = load_iris(as_frame=True)
    df = iris.frame

    st.write("Here's a preview of the dataset:")
    st.dataframe(df.head(), use_container_width=True)

    st.sidebar.header("Plot Controls")
    hist_feature = st.sidebar.selectbox("📈 Histogram Feature", iris.feature_names)
    scatter_x = st.sidebar.selectbox("X-axis", iris.feature_names, index=0)
    scatter_y = st.sidebar.selectbox("Y-axis", iris.feature_names, index=1)

    # Histogram
    st.write(f"### Histogram of **{hist_feature}**")
    fig1, ax1 = plt.subplots()
    ax1.hist(df[hist_feature], bins=15, color="#4a90e2", edgecolor="black")
    ax1.set_xlabel(hist_feature)
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # Scatter Plot
    st.write(f"### Scatter Plot: **{scatter_y} vs {scatter_x}**")
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(df[scatter_x], df[scatter_y], c=iris.target, cmap="viridis")
    ax2.set_xlabel(scatter_x)
    ax2.set_ylabel(scatter_y)
    st.pyplot(fig2)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#888;'>© 2025 · Pranav Datta Y Sharma · All Rights Reserved</p>", unsafe_allow_html=True)
