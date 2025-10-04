# Student Academic Performance Predictor

A full‑stack Data Science project that predicts student performance from features like attendance, test scores, study hours, and parental education.

## Stack
- **Frontend:** Streamlit (quick UI). You can swap to React later.
- **Backend:** Flask (REST API) + CORS
- **ML:** scikit‑learn (RandomForest) in a Pipeline with OneHotEncoder
- **DB (optional):** SQLite (store prediction history)

## Project Structure
```
student-performance-predictor/
├─ backend/
│  ├─ app.py                  # Flask API
│  ├─ requirements.txt        # Backend deps
│  └─ models/
│     └─ model.pkl            # Saved ML pipeline (created after training)
├─ frontend/
│  └─ streamlit_app.py        # Streamlit UI
├─ data/
│  ├─ student_training.csv    # With Final_Result (for training)
│  └─ predict_sample.csv      # Without Final_Result (for prediction)
└─ notebooks/
   └─ train_model.py          # Train and export model.pkl
```

---

## 1) Create a Python env (recommended)
```bash
cd student-performance-predictor
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

## 2) Install **backend** deps
```bash
pip install -r backend/requirements.txt
```

## 3) Train the model
Run the training script. It reads `data/student_training.csv` and writes `backend/models/model.pkl`.

```bash
python notebooks/train_model.py
```

## 4) Start the Flask API
```bash
python backend/app.py
```
The API runs at `http://127.0.0.1:5000`.

### Endpoints
- `GET /health` → API status
- `POST /predict` → JSON body: list of records with columns:
  `Attendance, Hours_Studied, Previous_Score, Parent_Education, Test1, Test2`
- `POST /predict-file` → multipart file upload (CSV)

## 5) Start the Streamlit UI
Open a new terminal (keep API running), activate the env again, then:
```bash
streamlit run frontend/streamlit_app.py
```
If your API is on a different host/port, set it in the sidebar.

## 6) (Optional) Save predictions to SQLite
The API will auto‑create `students.db` and append predictions to the `predictions` table when using `/predict-file`.

---

## Tips
- Replace the sample CSV with your real dataset (keep column names).
- Tweak the model in `notebooks/train_model.py` (switch to XGBoost, add CV, tune params).
- Swap Streamlit for React later by replacing the `frontend` folder.

Happy building! 🚀
