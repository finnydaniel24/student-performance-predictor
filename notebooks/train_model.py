
import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_training.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

target_col = "Final_Result"
y = df[target_col]
X = df.drop(columns=[target_col])

categorical = ["Parent_Education"]
numeric = [c for c in X.columns if c not in categorical]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ]
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", clf)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training...")
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="weighted")

print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Classification report:")
print(classification_report(y_test, preds))
print("Confusion matrix:")
print(confusion_matrix(y_test, preds))

# Save model pipeline
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(pipe, model_path)
print("Saved model to:", model_path)
