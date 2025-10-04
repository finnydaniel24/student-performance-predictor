
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("backend/models/model.pkl")

EXPECTED_COLUMNS = [
    "Attendance", "Hours_Studied", "Previous_Score", "Parent_Education", "Test1", "Test2"
]

def ensure_expected_columns(df: pd.DataFrame):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[EXPECTED_COLUMNS]

@app.route("/predict-file", methods=["POST"])
def predict_file():
    file = request.files["file"]
    df_raw = pd.read_csv(file)

    # normalize column names (robust Name handling)
    df_raw.columns = [c.strip().capitalize() if c.lower().strip()=="name" else c.strip() for c in df_raw.columns]

    feats = ensure_expected_columns(df_raw.copy())
    preds = model.predict(feats)
    conf = model.predict_proba(feats).max(axis=1) if hasattr(model,"predict_proba") else [None]*len(preds)

    out = df_raw.copy()
    out["Predicted_Performance"] = preds
    out["Confidence"] = conf

    return jsonify({
        "rows": len(out),
        "columns": list(out.columns),
        "data": out.to_dict(orient="records")
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df_raw = pd.DataFrame(data)
    df_raw.columns = [c.strip().capitalize() if c.lower().strip()=="name" else c.strip() for c in df_raw.columns]

    feats = ensure_expected_columns(df_raw.copy())
    preds = model.predict(feats)
    probs = model.predict_proba(feats).max(axis=1) if hasattr(model,"predict_proba") else [None]*len(preds)

    out = df_raw.copy()
    out["Predicted_Performance"] = preds
    out["Confidence"] = probs

    return jsonify({"data": out.to_dict(orient="records")})

if __name__ == "__main__":
    app.run(debug=True)
