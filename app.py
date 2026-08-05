import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


@st.cache_data
def load_artifacts():
    base = Path(__file__).parent
    model_path = base / "model.pkl"
    scaler_path = base / "scaler.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal_dimension_error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal_dimension'
]


def predict_dataframe(df, model, scaler):
    # Accept dataframes with proper columns or without names (use first 30 cols)
    if list(df.columns) != FEATURE_NAMES:
        if df.shape[1] >= 30:
            X = df.iloc[:, :30].values
        else:
            raise ValueError("Input must have at least 30 features (columns).")
    else:
        X = df[FEATURE_NAMES].values

    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    # handle keras-style probabilistic outputs
    if preds.ndim > 1 and preds.shape[1] > 1:
        probs = preds
        labels = np.argmax(probs, axis=1)
        confidences = probs.max(axis=1)
    else:
        probs = preds.ravel()
        labels = (probs > 0.5).astype(int)
        confidences = np.where(labels == 1, probs, 1 - probs)

    # Map: 1 = benign, 0 = malignant (as used in the notebook)
    label_map = {1: "Benign", 0: "Malignant"}
    results = pd.DataFrame({
        "prediction": [label_map.get(int(l), str(l)) for l in labels],
        "confidence": confidences
    })
    return results


def single_input_form():
    st.write("Provide feature values (30 features). You can also upload a CSV with the same columns.")
    cols = st.columns(3)
    inputs = []
    for i, name in enumerate(FEATURE_NAMES):
        col = cols[i % 3]
        val = col.number_input(name, value=0.0, format="%.6f")
        inputs.append(val)
    return np.array(inputs).reshape(1, -1)


def main():
    st.title("Can-Cure — Breast Cancer Prediction")
    st.write("Upload a CSV of samples or fill a single sample below and click Predict.")

    model, scaler = load_artifacts()

    uploaded = st.file_uploader("Upload CSV file with features (first 30 columns used)", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head())
            if st.button("Predict uploaded data"):
                results = predict_dataframe(df, model, scaler)
                out = pd.concat([df.reset_index(drop=True).iloc[:, :30], results], axis=1)
                out = out.rename(columns={"prediction": "Diagnosis", "confidence": "Confidence"})
                st.write("Results — Diagnosis")
                # show diagnosis and confidence clearly
                st.dataframe(out[["Diagnosis", "Confidence"]])
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

    st.markdown("---")
    st.subheader("Single sample prediction")
    with st.form("single_form"):
        sample = single_input_form()
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            df_sample = pd.DataFrame(sample, columns=FEATURE_NAMES)
            res = predict_dataframe(df_sample, model, scaler)
            # Display clear diagnosis and confidence
            diagnosis = res.loc[0, "prediction"]
            confidence = res.loc[0, "confidence"]
            st.subheader(f"Diagnosis: {diagnosis}")
            st.write(f"Confidence: {confidence:.3f}")
            if diagnosis == "Malignant":
                st.error("Model predicts a malignant tumor — please consult a medical professional.")
            else:
                st.success("Model predicts a benign tumor.")
        except Exception as e:
            st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
