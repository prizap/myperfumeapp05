# app_streamlit_parfum_light.py
"""
Streamlit app for Perfume Situation classification (Day / Night / Versatile).

Features:
- Load pipeline (.joblib) from path or upload
- Single prediction (manual input)
- Batch prediction via CSV upload and download results
- Show prediction probabilities if available

Keep this app minimal for lighter deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Klasifikasi Penggunaan Parfum (Day/Night/Versatile)", layout="centered")

# ---------------------------
# Minimal SafeSVD (keamanan unpickle)
# ---------------------------
class SafeSVD(BaseEstimator, TransformerMixin):
    """Minimal SafeSVD used during training; included so joblib can unpickle pipelines that used it."""
    def __init__(self, n_components=50, random_state=42):
        self.n_components = int(n_components)
        self.random_state = random_state
        self.svd_ = None
        self.do_reduce_ = False

    def fit(self, X, y=None):
        try:
            n_features = X.shape[1]
        except Exception:
            X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            n_features = X_arr.shape[1]
        if n_features > 1 and self.n_components < n_features:
            n_comp = min(self.n_components, n_features - 1)
            if n_comp >= 1:
                self.svd_ = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
                self.svd_.fit(X)
                self.do_reduce_ = True
        return self

    def transform(self, X):
        if self.do_reduce_ and self.svd_ is not None:
            return self.svd_.transform(X)
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

# ---------------------------
# Utils: normalize text and build small dataframe for single prediction
# ---------------------------
def normalize_text(s: str) -> str:
    s = '' if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r'[\/;|•]+', ',', s)
    s = re.sub(r'\s*[,\;]\s*', ', ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def build_single_df(top_notes, mid_notes, base_notes, brand, concentrate, gender, price, size):
    df = pd.DataFrame([{
        'top notes': top_notes,
        'mid notes': mid_notes,
        'base notes': base_notes,
        'brand': brand,
        'concentrate': concentrate,
        'gender': gender,
        'price': price,
        'size': size
    }])
    for c in ['top notes','mid notes','base notes','brand','concentrate','gender']:
        df[c] = df[c].fillna('').astype(str).apply(normalize_text)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    df['size']  = pd.to_numeric(df['size'], errors='coerce').fillna(0.0)
    df['all_notes'] = (df['top notes'] + ', ' + df['mid notes'] + ', ' + df['base notes']).str.replace(r'\s+', ' ', regex=True).str.strip(', ')
    return df

# ---------------------------
# Sidebar: model load
# ---------------------------
st.sidebar.header("Model")
default_model_path = "full_pipeline_situation_model.joblib"
model_path = st.sidebar.text_input("Model path (.joblib)", value=default_model_path)
uploaded_model = st.sidebar.file_uploader("Or upload pipeline (.joblib)", type=["joblib","pkl"])
show_prob = st.sidebar.checkbox("Show prediction probabilities (if available)", value=True)

def load_model(path_text, uploaded_file):
    model = None
    if uploaded_file is not None:
        try:
            model = joblib.load(uploaded_file)
            st.sidebar.success("Model loaded from upload.")
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded model: {e}")
            model = None
    else:
        if os.path.exists(path_text):
            try:
                model = joblib.load(path_text)
                st.sidebar.success(f"Model loaded from path: {path_text}")
            except Exception as e:
                st.sidebar.error(f"Failed to load model from path: {e}")
                model = None
        else:
            st.sidebar.info("No model path found; upload .joblib or set path.")
    return model

model = load_model(model_path, uploaded_model)

# ---------------------------
# Main UI: single prediction
# ---------------------------
st.title("Klasifikasi Penggunaan Parfum (Day / Night / Versatile)")
st.markdown("Prediksi penggunaan parfum apakah cocok untuk kondisi **Day / Night / Versatile**.")

st.subheader("Single prediction")
col1, col2 = st.columns([2,1])
with col1:
    top_notes = st.text_area("Top notes", value="", height=80)
    mid_notes = st.text_area("Mid notes", value="", height=80)
    base_notes = st.text_area("Base notes", value="", height=80)
with col2:
    brand = st.text_input("Brand", value="")
    concentrate = st.text_input("Concentrate (e.g., edp, xdp)", value="")
    gender = st.selectbox("Gender", options=["unisex","female","male",""], index=0)
    price = st.number_input("Price", min_value=0.0, value=0.0, step=1000.0)
    size = st.number_input("Size (ml)", min_value=0, value=50, step=1)

if st.button("Predict single"):
    sample_df = build_single_df(top_notes, mid_notes, base_notes, brand, concentrate, gender, price, size)
    if model is None:
        st.error("Model not loaded. Upload or provide path in sidebar.")
    else:
        try:
            X_input = sample_df[['all_notes','brand','concentrate','gender','price','size']]
            pred = model.predict(X_input)
            st.success(f"Predicted situation: **{pred[0]}**")
            if show_prob and hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)[0]
                classes = list(model.classes_)
                prob_df = pd.DataFrame({"class": classes, "probability": probs})
                st.table(prob_df.sort_values("probability", ascending=False).reset_index(drop=True))
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ---------------------------
# Batch prediction (CSV)
# ---------------------------
st.subheader("Batch prediction (CSV)")
st.markdown("Upload CSV with columns: `top notes`, `mid notes`, `base notes`, `brand`, `concentrate`, `gender`, `price`, `size`")
uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="batch_light")

if uploaded_csv is not None:
    try:
        df_batch = pd.read_csv(uploaded_csv)
        st.write("Preview:")
        st.dataframe(df_batch.head(5))
        required = ['top notes','mid notes','base notes','brand','concentrate','gender','price','size']
        missing = [c for c in required if c not in df_batch.columns]
        if missing:
            st.error(f"CSV missing required columns: {missing}")
        else:
            for c in ['top notes','mid notes','base notes','brand','concentrate','gender']:
                df_batch[c] = df_batch[c].fillna('').astype(str).apply(normalize_text)
            df_batch['price'] = pd.to_numeric(df_batch['price'], errors='coerce').fillna(0.0)
            df_batch['size']  = pd.to_numeric(df_batch['size'], errors='coerce').fillna(0.0)
            df_batch['all_notes'] = (df_batch['top notes'] + ', ' + df_batch['mid notes'] + ', ' + df_batch['base notes']).str.replace(r'\s+', ' ', regex=True).str.strip(', ')
            if model is None:
                st.error("Model not loaded. Upload or provide path in sidebar.")
            else:
                try:
                    X_in = df_batch[['all_notes','brand','concentrate','gender','price','size']]
                    preds = model.predict(X_in)
                    df_batch['predicted_situation'] = preds
                    if show_prob and hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_in)
                        df_batch['pred_prob_max'] = np.max(probs, axis=1)
                        for i, cls in enumerate(model.classes_):
                            df_batch[f"prob_{cls}"] = probs[:, i]
                    st.success("Batch prediction done.")
                    st.dataframe(df_batch.head(10))
                    csv_out = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button("Download results CSV", csv_out, file_name="predictions_light.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Error during batch prediction: {e}")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("powered by Streamlit. Developed for perfume situation classification.")