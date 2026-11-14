import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

st.set_page_config(page_title="AI Placement Predictor & ATS Scanner", layout="wide")
st.title("AI-Powered Student Placement Predictor & ATS Resume Scanner")
st.caption("Developed by Vishal Pandey | Python, Scikit-Learn Random Forest, Streamlit & Gemini API Ready")

@st.cache_data
def load_and_train_model():
    np.random.seed(42)
    n = 300
    cgpa = np.round(np.random.uniform(5.5, 9.8, n), 2)
    internships = np.random.choice([0, 1, 2, 3], n, p=[0.3, 0.4, 0.2, 0.1])
    projects = np.random.choice([1, 2, 3, 4, 5], n, p=[0.15, 0.35, 0.3, 0.15, 0.05])
    aptitude = np.random.randint(45, 99, n)
    coding_score = np.random.randint(40, 100, n)
    backlogs = np.random.choice([0, 1, 2], n, p=[0.75, 0.18, 0.07])
    
    score = (cgpa * 8) + (internships * 12) + (projects * 6) + (aptitude * 0.3) + (coding_score * 0.4) - (backlogs * 18)
    placed = (score > 118).astype(int)
    
    df = pd.DataFrame({
        "CGPA": cgpa, "Internships": internships, "Projects": projects,
        "Aptitude_Score": aptitude, "Coding_Score": coding_score,
        "Active_Backlogs": backlogs, "Placed": placed
    })
    
    X = df.drop("Placed", axis=1)
    y = df["Placed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc, df

clf, model_acc, sample_df = load_and_train_model()

tab1, tab2 = st.tabs(["1. ML Placement Probability Predictor", "2. ATS Resume Keyword & Match Scanner"])

with tab1:
    st.subheader(f"Random Forest Placement Classifier (Validation Accuracy: {model_acc*100:.1f}%)")
    col1, col2 = st.columns(2)
    with col1:
        u_cgpa = st.slider("Current CGPA", 5.0, 10.0, 8.1, 0.1)
        u_intern = st.number_input("Completed Internships", 0, 5, 2)
        u_proj = st.number_input("Technical Projects Built", 1, 8, 3)
    with col2:
        u_apt = st.slider("Quantitative Aptitude Score (out of 100)", 30, 100, 82)
        u_code = st.slider("Technical / Coding Assessment Score (out of 100)", 30, 100, 78)
        u_back = st.selectbox("Active Academic Backlogs", [0, 1, 2, 3], index=0)
        
    if st.button("Predict Placement Probability"):
        features = pd.DataFrame([[u_cgpa, u_intern, u_proj, u_apt, u_code, u_back]],
                                columns=["CGPA", "Internships", "Projects", "Aptitude_Score", "Coding_Score", "Active_Backlogs"])
        prob = clf.predict_proba(features)[0][1] * 100
        st.metric("Predicted Placement Probability", f"{prob:.1f}%")
        if prob >= 70:
            st.success("Strong Profile! High probability of clearing technical & quantitative shortlists.")
        elif prob >= 45:
            st.warning("Moderate Profile. Adding 1 more deployed project or improving coding metrics will boost odds.")
        else:
            st.error("At-Risk Profile. Focus on clearing backlogs and building hands-on portfolio projects.")

with tab2:
    st.subheader("ATS Resume Keyword Scanner & Gemini AI Integration")
    jd_text = st.text_area("Paste Target Job Description (JD)", 
                           "Looking for a Data Analyst / Quantitative Intern skilled in Python, SQL, Power BI, Excel, Pandas, Machine Learning, Backtesting, and Statistics.")
    resume_text = st.text_area("Paste Your Resume Text", 
                               "Vishal Pandey. B.Tech CSE student skilled in Python, SQL, Power BI, Advanced Excel, Pandas, Scikit-Learn Machine Learning, and Pine Script Algorithmic Backtesting.")
    if st.button("Run ATS Scan"):
        stop_words = {"looking", "for", "a", "in", "and", "the", "to", "with", "of", "or", "student", "skilled"}
        jd_words = {w.lower() for w in re.findall(r"[A-Za-z0-9#+]+", jd_text) if w.lower() not in stop_words and len(w) > 2}
        res_words = {w.lower() for w in re.findall(r"[A-Za-z0-9#+]+", resume_text)}
        matched = sorted(list(jd_words.intersection(res_words)))
        missing = sorted(list(jd_words - res_words))
        score = (len(matched) / max(len(jd_words), 1)) * 100
        st.metric("ATS Keyword Match Score", f"{score:.1f}%")
        st.write("**Matched Keywords:**", ", ".join(matched) if matched else "None")
        st.write("**Missing Keywords to Add:**", ", ".join(missing) if missing else "None - 100% Match!")
