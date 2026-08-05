# Can-Cure — Breast Cancer Predictor

This repository contains a Streamlit frontend for a breast cancer prediction model trained on scikit-learn's built-in `breast_cancer` dataset (from `sklearn.datasets`). The model predicts whether a tumor is benign or malignant based on 30 numeric features derived from digitized images of fine needle aspirate (FNA) of breast masses.

![Output image](output.png)

**How it works**
- The model expects 30 numeric features (the same features as `sklearn.datasets.load_breast_cancer`).
- The app loads `model.pkl` and `scaler.pkl` from the project root to preprocess inputs and produce predictions.
- Predictions are shown as `Benign` or `Malignant` with a confidence score.

**Quick Start**

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# On macOS / Linux
source .venv/bin/activate
# On Windows PowerShell
.venv\\Scripts\\Activate.ps1
# Or on Windows CMD
.venv\\Scripts\\activate.bat
pip install -r requirements_app.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Usage in the UI:
- Upload a CSV file where the first 30 columns correspond to the breast cancer features (an example file `example_input.csv` is provided).
- Or fill the single-sample form (30 number inputs) and click `Predict`.

**Interpreting results**
- `Benign` — model predicts non-cancerous tumor.
- `Malignant` — model predicts cancerous tumor.
- `confidence` — probability/confidence score for the predicted label.

**Example files**
- `app.py`: Streamlit frontend and prediction logic.
- `model.ipynb`: Notebook used for training and analysis.
- `requirements_app.txt`: Python dependencies.
- `example_input.csv`: Example CSV with 30 features (ready for upload).
- `output.png`: Example visualization produced by the app.

**Dataset reference**
- scikit-learn `breast_cancer` dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

**Developer**: Surya Dutta


