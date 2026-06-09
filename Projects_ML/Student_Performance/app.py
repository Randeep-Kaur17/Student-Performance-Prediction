import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load Model
model = joblib.load("student_performance_model.pkl")

# Title
st.title("🎓 Student Performance Prediction System")
st.markdown("Predict a student's performance based on demographic and academic factors.")

st.divider()

# Sidebar
st.sidebar.header("Student Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["female", "male"]
)

race = st.sidebar.selectbox(
    "Race / Ethnicity",
    [
        "group A",
        "group B",
        "group C",
        "group D",
        "group E"
    ]
)

parent_education = st.sidebar.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)

lunch = st.sidebar.selectbox(
    "Lunch Type",
    [
        "standard",
        "free/reduced"
    ]
)

prep_course = st.sidebar.selectbox(
    "Test Preparation Course",
    [
        "none",
        "completed"
    ]
)

reading_score = st.sidebar.slider(
    "Reading Score",
    0,
    100,
    60
)

writing_score = st.sidebar.slider(
    "Writing Score",
    0,
    100,
    60
)

# Main Area
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Reading Score",
        reading_score
    )

with col2:
    st.metric(
        "Writing Score",
        writing_score
    )

# Prediction Button
if st.button("Predict Math Score", use_container_width=True):

    input_data = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [parent_education],
        "lunch": [lunch],
        "test preparation course": [prep_course],
        "reading score": [reading_score],
        "writing score": [writing_score]
    })

    prediction = model.predict(input_data)

    st.success(
        f"🎯 Predicted Math Score: {prediction[0]:.2f}"
    )

    if prediction[0] >= 80:
        st.balloons()
        st.info("Excellent Performance Expected ⭐")

    elif prediction[0] >= 60:
        st.info("Good Performance Expected 👍")

    else:
        st.warning("Needs More Practice 📚")

st.divider()

st.subheader("About Project")

st.write("""
This Machine Learning project predicts a student's Math Score using:

- Gender
- Race/Ethnicity
- Parental Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

Algorithm Used:
- Random Forest Regressor

Developed using:
- Python
- Scikit-Learn
- Pandas
- Streamlit
""")