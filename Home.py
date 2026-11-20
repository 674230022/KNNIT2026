
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# ส่วนที่ 1 ตั้งค่าหน้าเว็บไซต์
# -----------------------------

st.set_page_config(page_title="Heart Disease KNN", layout="wide")

st.header("โปรเจคการจำแนกข้อมูลโรคหัวใจด้วย KNN")

st.write("ระบบจำแนกข้อมูลโรคหัวใจโดยใช้ Machine Learning")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("โครงสร้างหัวใจ")
    st.image("./img/heart_structure.jpg")

with col2:
    st.header("คลื่นไฟฟ้าหัวใจ (ECG)")
    st.image("./img/ecg.jpg")

with col3:
    st.header("สุขภาพหัวใจ")
    st.image("./img/heart_health.jpg")

# -----------------------------
# ส่วนที่ 2 อ่านข้อมูล
# -----------------------------

dt = pd.read_csv("./Heart3.csv")

st.subheader("สถิติข้อมูลโรคหัวใจ")
st.write(dt.head(10))

# -----------------------------
# ส่วนที่ 3 แสดงกราฟข้อมูล
# -----------------------------

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px">
<center><h5>สถิติข้อมูลโรคหัวใจ</h5></center>
</div>
"""

st.markdown(html_7, unsafe_allow_html=True)

# นับจำนวนผู้ที่มีและไม่มีโรคหัวใจ

dt1 = dt["HeartDisease"].value_counts().sort_index()

dt1.index = ["ไม่มีโรคหัวใจ (0)", "มีโรคหัวใจ (1)"]

if st.checkbox("แสดงการจินตทัศน์ข้อมูล"):
    st.bar_chart(dt1)

# -----------------------------
# ส่วนที่ 4 เตรียมข้อมูล KNN
# -----------------------------

X = dt.drop("HeartDisease", axis=1)
y = dt["HeartDisease"]

# แบ่งข้อมูล Train และ Test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# สร้างโมเดล KNN พร้อมปรับมาตรฐานข้อมูล

Knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# ฝึกโมเดล

Knn_model.fit(X_train, y_train)

# ทดสอบโมเดล

y_pred = Knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("ประสิทธิภาพโมเดล KNN")

st.metric("Accuracy", f"{accuracy * 100:.2f}%")

# -----------------------------
# ส่วนที่ 5 ทำนายข้อมูลโรคหัวใจ
# -----------------------------

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px">
<center><h5>ทำนายข้อมูลโรคหัวใจ</h5></center>
</div>
"""

st.markdown(html_8, unsafe_allow_html=True)

st.write("กรุณากรอกข้อมูลเพื่อทำนายผล")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("อายุ (Age)", 1, 100, 50)

    sex = st.selectbox(
        "เพศ (Sex)",
        sorted(dt["Sex"].unique())
    )

    chest = st.selectbox(
        "ประเภทอาการเจ็บหน้าอก (ChestPainType)",
        sorted(dt["ChestPainType"].unique())
    )

    restingbp = st.number_input(
        "ความดันโลหิตขณะพัก (RestingBP)",
        0, 300, 120
    )

with col2:
    cholesterol = st.number_input(
        "คอเลสเตอรอล (Cholesterol)",
        0, 1000, 200
    )

    fastingbs = st.selectbox(
        "FastingBS",
        sorted(dt["FastingBS"].unique())
    )

    restingecg = st.selectbox(
        "RestingECG",
        sorted(dt["RestingECG"].unique())
    )

    maxhr = st.number_input(
        "อัตราการเต้นหัวใจสูงสุด (MaxHR)",
        1, 300, 150
    )

with col3:
    exercise = st.selectbox(
        "ExerciseAngina",
        sorted(dt["ExerciseAngina"].unique())
    )

    oldpeak = st.number_input(
        "Oldpeak",
        value=0.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "ST_Slope",
        sorted(dt["ST_Slope"].unique())
    )

# -----------------------------
# ส่วนที่ 6 แสดงผลการทำนาย
# -----------------------------

if st.button("ทำนายผล"):

    x_input = pd.DataFrame([[
        age,
        sex,
        chest,
        restingbp,
        cholesterol,
        fastingbs,
        restingecg,
        maxhr,
        exercise,
        oldpeak,
        st_slope
    ]], columns=X.columns)

    out = Knn_model.predict(x_input)

    st.subheader("ผลการทำนาย")

    if out[0] == 1:
        st.error("ผลการทำนาย: มีโรคหัวใจ (HeartDisease = 1)")

    else:
        st.success("ผลการทำนาย: ไม่มีโรคหัวใจ (HeartDisease = 0)")

    # แสดงค่าความน่าจะเป็นจากโมเดล

    probability = Knn_model.predict_proba(x_input)[0]

    st.write("ความน่าจะเป็นของผลการทำนาย")

    st.write(f"กลุ่ม 0: {probability[0] * 100:.2f}%")

    st.write(f"กลุ่ม 1: {probability[1] * 100:.2f}%")

    st.caption(
        "ผลลัพธ์เป็นการทำนายจากโมเดลเพื่อการศึกษา "
        "ไม่ใช่การวินิจฉัยทางการแพทย์"
    )

# -----------------------------

# -----------------------------
# ส่วนที่ 7 ข้อมูลผู้จัดทำ
# -----------------------------

st.markdown("---")

st.markdown(
    """
    <h3 style="text-align:center; margin-bottom:20px;">
        ผู้จัดทำโครงงาน
    </h3>
    """,
    unsafe_allow_html=True
)

with st.container(border=True):
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        left, center, right = st.columns([1, 2, 1])

        with center:
            st.image(
                "./img/Profile.jpg",
                width=180
            )

    st.markdown(
        """
        <div style="text-align:center; padding:5px;">
            <h4 style="margin-bottom:5px;">
                ผู้จัดทำโครงงาน
            </h4>
            <p style="color:gray; margin:0;">
                สาขาเทคโนโลยีสารสนเทศ
            </p>
            <p style="color:gray; font-size:13px;">
                KNN Heart Disease Prediction
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
