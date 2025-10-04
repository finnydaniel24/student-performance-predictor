
import streamlit as st
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# ---- Header (no logo) ----
st.title("🎓 Student Academic Performance Predictor")

# Sidebar (no logo, no uploader)
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("♻️ Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ App reset successfully! Reloading...")
        st.rerun()

    api_url = st.text_input("Flask API Base URL", value="http://127.0.0.1:5000")
    st.markdown("Endpoints: `/predict`, `/predict-file`, `/student/<name>`")

tab1, tab2 = st.tabs([
    "📄 Predict from CSV",
    "🧍 Predict Single Student"
])

# ---------- Class PDF GENERATOR (summary only, no logo) ----------
def generate_pdf_report(dataframe):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title + timestamp
    pdf.setTitle("Student Performance Report")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(150, height - 60, "Student Performance Report")
    pdf.setFont("Helvetica", 9)
    pdf.drawRightString(width - 40, height - 60, datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Summary
    pdf.setFont("Helvetica", 12)
    pdf.drawString(40, height - 100, f"Total Students: {len(dataframe)}")

    # Counts
    pdf.setFont("Helvetica", 11)
    y = height - 130
    summary = dataframe["Predicted_Performance"].value_counts().to_dict()
    for category, count in summary.items():
        pdf.drawString(40, y, f"{category}: {count}")
        y -= 16

    # --- Chart: Performance Distribution (no specific colors) ---
    plt.figure(figsize=(5, 3))
    dataframe["Predicted_Performance"].value_counts().plot(kind="bar")
    plt.title("Performance Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    chart_path = "performance_plot.png"
    plt.savefig(chart_path)
    plt.close()
    try:
        pdf.drawImage(chart_path, 60, y - 210, width=420, height=200, preserveAspectRatio=True)
    except Exception:
        pass

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer

# ---------- Rule-based Summary for a single student ----------
def rule_based_summary(student: dict) -> str:
    def get_num(k, default=0.0):
        try:
            return float(student.get(k, default))
        except Exception:
            return default

    name = str(student.get("Name", "This student"))
    att = get_num("Attendance")
    hrs = get_num("Hours_Studied")
    t1 = get_num("Test1")
    t2 = get_num("Test2")
    prev = get_num("Previous_Score")
    avg_test = (t1 + t2) / 2 if (t1 or t2) else 0.0
    perf = str(student.get("Predicted_Performance", "Unknown"))

    parts = []

    # Attendance insight
    if att >= 90:
        parts.append("excellent attendance")
    elif att >= 75:
        parts.append("good attendance")
    elif att >= 60:
        parts.append("moderate attendance")
    else:
        parts.append("low attendance")

    # Study hours
    if hrs >= 10:
        parts.append("strong weekly study time")
    elif hrs >= 6:
        parts.append("decent weekly study time")
    elif hrs >= 3:
        parts.append("limited weekly study time")
    else:
        parts.append("very low weekly study time")

    # Tests insight
    if avg_test >= 80:
        parts.append("high recent test scores")
    elif avg_test >= 65:
        parts.append("solid recent test scores")
    elif avg_test >= 50:
        parts.append("mixed recent test scores")
    else:
        parts.append("low recent test scores")

    # Previous score
    if prev >= 80:
        hist = "a strong prior record"
    elif prev >= 60:
        hist = "an average prior record"
    else:
        hist = "a weak prior record"

    # Compose paragraph
    sentence1 = f"{name} shows {parts[0]} with {parts[1]}, and {parts[2]}. Past performance indicates {hist}."
    sentence2 = f"The predicted category is '{perf}'"
    conf = student.get("Confidence", None)
    if conf is not None:
        try:
            sentence2 += f" with confidence {float(conf):.2f}."
        except Exception:
            sentence2 += "."
    else:
        sentence2 += "."

    # Recommendation
    if perf.lower() in ["good", "excellent"] or (att >= 85 and avg_test >= 75):
        rec = "Maintaining consistent study habits and attendance should sustain current performance."
    elif perf.lower() in ["average"] or (avg_test >= 55):
        rec = "Targeted practice on weaker topics and a small boost in study hours could elevate results."
    else:
        rec = "A structured study plan and improved attendance are recommended to enhance outcomes."

    return f"{sentence1} {sentence2} {rec}"

# ---------- Individual PDF generator (summary at the bottom, no logo) ----------
def generate_single_student_report(student):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title + timestamp
    pdf.setTitle("Individual Student Report")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(170, height - 60, "Individual Student Report")
    pdf.setFont("Helvetica", 9)
    pdf.drawRightString(width - 40, height - 60, datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Top info
    pdf.setFont("Helvetica", 12)
    name = str(student.get("Name", "Unknown"))
    perf = str(student.get("Predicted_Performance", "N/A"))
    conf = student.get("Confidence", None)
    conf_txt = f"{float(conf):.2f}" if conf is not None else "N/A"
    pdf.drawString(40, height - 100, f"Name: {name}")
    pdf.drawString(40, height - 120, f"Predicted Performance: {perf}")
    pdf.drawString(40, height - 140, f"Confidence: {conf_txt}")

    # Attributes
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, height - 165, "Attributes:")
    pdf.setFont("Helvetica", 11)
    y = height - 185
    for key in ["Attendance", "Hours_Studied", "Previous_Score", "Parent_Education", "Test1", "Test2"]:
        if key in student:
            pdf.drawString(60, y, f"{key}: {student.get(key)}")
            y -= 16

    # Chart of test scores (no specific colors)
    test_cols = [k for k in ["Test1", "Test2"] if k in student]
    if test_cols:
        vals = []
        for c in test_cols:
            try:
                vals.append(float(student.get(c, 0)))
            except Exception:
                vals.append(0.0)
        plt.figure(figsize=(4, 2.2))
        plt.bar(test_cols, vals)
        plt.title("Test Scores")
        plt.ylabel("Score")
        plt.ylim(0, 100)
        plt.tight_layout()
        chart_path = "individual_chart.png"
        plt.savefig(chart_path)
        plt.close()
        try:
            pdf.drawImage(chart_path, 40, y - 180, width=300, height=160, preserveAspectRatio=True)
        except Exception:
            pass
        y = y - 190

    # AI summary at the bottom
    summary_text = rule_based_summary(student)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y - 10, "AI Summary:")
    pdf.setFont("Helvetica", 11)

    # Wrap text manually
    import textwrap as tw
    wrapped = tw.wrap(summary_text, width=95)
    yy = y - 28
    for line in wrapped:
        if yy < 60:
            pdf.showPage()
            yy = height - 60
            pdf.setFont("Helvetica", 11)
        pdf.drawString(40, yy, line)
        yy -= 15

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer

# --- TAB 1 ---
with tab1:
    file = st.file_uploader("Upload CSV", type=["csv"], key="csv_file")

    if file is not None:
        if st.session_state.get("uploaded_filename") != file.name:
            st.session_state["uploaded_filename"] = file.name
            st.session_state["uploaded_df"] = pd.read_csv(file)
            st.session_state.pop("predicted_data", None)

    if "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"]
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        run_pred = st.button("🔮 Predict for CSV")
        if run_pred or "predicted_data" in st.session_state:
            if "predicted_data" not in st.session_state:
                try:
                    file.seek(0)
                    files = {"file": (file.name, file.getvalue(), "text/csv")}
                    res = requests.post(f"{api_url}/predict-file", files=files, timeout=120)
                    if res.status_code == 200:
                        payload = res.json()
                        out = pd.DataFrame(payload["data"])
                        st.session_state["predicted_data"] = out
                        st.success("✅ Prediction Completed!")
                    else:
                        st.error(f"API error: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    if "predicted_data" in st.session_state:
        out = st.session_state["predicted_data"].copy()
        if "Name" in out.columns:
            out["Name"] = out["Name"].astype(str).str.strip()

        st.subheader("📊 Predicted Results")
        st.dataframe(out)

        selected_row = None

        if "Name" in out.columns:
            st.subheader("🔤 Filter by Name (type-ahead)")
            q = st.text_input("Start typing a name…").strip().lower()
            filtered = out[out["Name"].str.lower().str.contains(q)] if q else out
            st.write(f"Showing {len(filtered)}/{len(out)} rows")
            st.dataframe(filtered)

            if not filtered.empty:
                pick = st.selectbox("Pick from filtered results:", filtered["Name"].unique().tolist(), key="pick_from_filtered")
                sel = filtered[filtered["Name"].str.lower() == str(pick).lower()]
                if not sel.empty:
                    st.write("### 🧍 Student Performance")
                    st.dataframe(sel)
                    selected_row = sel.iloc[0].to_dict()
                    try:
                        conf_val = float(selected_row.get("Confidence", 0.0))
                        st.info(f"**{selected_row['Name']}** → **{selected_row['Predicted_Performance']}** (Confidence: {conf_val:.2f})")
                    except Exception:
                        st.info(f"**{selected_row['Name']}** → **{selected_row['Predicted_Performance']}**")

        # Class CSV download
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv")

        # Class PDF (summary only)
        if st.button("📄 Generate Class PDF Report"):
            pdf_buffer = generate_pdf_report(out)
            st.download_button(
                label="⬇️ Download Class Report (PDF)",
                data=pdf_buffer,
                file_name="Class_Report.pdf",
                mime="application/pdf"
            )

        # Individual PDF (with rule-based AI summary at bottom)
        if selected_row is not None:
            if st.button(f"📄 Generate Individual Report for {selected_row['Name']}"):
                pdf_buffer = generate_single_student_report(selected_row)
                st.download_button(
                    label=f"⬇️ Download {selected_row['Name']}'s Report (PDF)",
                    data=pdf_buffer,
                    file_name=f"{selected_row['Name']}_Report.pdf",
                    mime="application/pdf"
                )

# --- TAB 2 ---
with tab2:
    with st.form("single_student_form"):
        Name = st.text_input("Name", value="John Doe")
        Attendance = st.number_input("Attendance (%)", 0, 100, 85)
        Hours_Studied = st.number_input("Hours Studied", 0, 20, 6)
        Previous_Score = st.number_input("Previous Score", 0, 100, 70)
        Parent_Education = st.selectbox("Parent Education", ["HighSchool","Bachelors","Masters","PhD"])
        Test1 = st.number_input("Test 1", 0, 100, 65)
        Test2 = st.number_input("Test 2", 0, 100, 72)
        submitted = st.form_submit_button("Predict")

    if submitted:
        data = [{
            "Name": Name,
            "Attendance": Attendance,
            "Hours_Studied": Hours_Studied,
            "Previous_Score": Previous_Score,
            "Parent_Education": Parent_Education,
            "Test1": Test1,
            "Test2": Test2
        }]
        try:
            res = requests.post(f"{api_url}/predict", json=data, timeout=60)
            if res.status_code == 200:
                result = res.json()["data"][0]
                st.success(f"**{result['Name']}** → **{result['Predicted_Performance']}** "
                           f"(Confidence: {float(result.get('Confidence', 0.0)):.2f})")
            else:
                st.error(f"API error: {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
