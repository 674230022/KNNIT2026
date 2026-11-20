
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="💜",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #171326;
    color: #F4F0FC;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, p, label {
    color: #F4F0FC;
}

[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    background: linear-gradient(120deg, #302448, #211A34);
    border: 1px solid #49385F;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero-tag {
    color: #BDA5F3;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 600;
}

.hero h1 {
    color: #FFFFFF;
    font-size: 35px;
    margin: 12px 0 5px 0;
}

.hero p {
    color: #C6BDD8;
    font-size: 15px;
}

.section-title {
    color: #D4C2FF;
    font-size: 21px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 15px;
    border-left: 4px solid #A78BFA;
    padding-left: 12px;
}

div[data-testid="stMetric"] {
    background: #28213B;
    border: 1px solid #45365B;
    padding: 18px;
    border-radius: 13px;
}

div[data-testid="stMetricLabel"] {
    color: #C7B9E4;
}

div[data-testid="stMetricValue"] {
    color: #D5C3FF;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #211B32;
    border: 1px solid #423451;
    border-radius: 15px;
}

.stButton > button {
    background: #7654B8;
    color: white;
    border: 1px solid #9676D0;
    border-radius: 10px;
    padding: 10px 25px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #9270D0;
    color: white;
    border-color: #BDA5F3;
}

.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
    background: #2B243C;
    color: white;
    border-color: #59476F;
    border-radius: 8px;
}

.stSlider [data-baseweb="slider"] {
    color: #B99AEF;
}

.stCheckbox label {
    color: #E0D5F4;
}

[data-testid="stDataFrame"] {
    border: 1px solid #493A5F;
    border-radius: 10px;
    overflow: hidden;
}

.stAlert {
    border-radius: 12px;
}

div[data-testid="stCaptionContainer"] {
    color: #AFA2C5;
}

hr {
    border-color: #493A5F;
}

.profile-card {
    background: linear-gradient(145deg, #29213D, #201A30);
    border: 1px solid #514064;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.profile-card h3 {
    color: #E5D8FF;
}

.profile-card p {
    color: #BDB1D0;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <div class="hero-tag">HEALTH ANALYTICS / 2026</div>
    <h1>Heart Disease Prediction</h1>
    <p>
        ระบบจำแนกข้อมูลโรคหัวใจด้วยเทคนิค
        K-Nearest Neighbors (KNN)
    </p>
</div>
""", unsafe_allow_html=True)


dt = pd.read_csv("./Heart3.csv")

X = dt.drop("HeartDisease", axis=1)
y = dt["HeartDisease"]

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Dataset Records", f"{len(dt):,}")

with m2:
    st.metric("Features", len(X.columns))

with m3:
    st.metric("Algorithm", "KNN (K=5)")


st.markdown(
    '<div class="section-title">Heart Health Gallery</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.image("./img/heart_structure.jpg", use_container_width=True)
        st.markdown("**Heart Anatomy**")
        st.caption("โครงสร้างและส่วนประกอบของหัวใจ")

with col2:
    with st.container(border=True):
        st.image("./img/ecg.jpg", use_container_width=True)
        st.markdown("**Electrocardiogram**")
        st.caption("คลื่นไฟฟ้าหัวใจ (ECG)")

with col3:
    with st.container(border=True):
        st.image("./img/heart_health.jpg", use_container_width=True)
        st.markdown("**Heart Wellness**")
        st.caption("การดูแลสุขภาพหัวใจ")


st.markdown(
    '<div class="section-title">Dataset Overview</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown("**ข้อมูลตัวอย่างจาก Heart3.csv**")
    st.dataframe(dt.head(10), use_container_width=True)


st.markdown(
    '<div class="section-title">Heart Disease Statistics</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    dt1 = dt["HeartDisease"].value_counts().sort_index()

    dt1.index = [
        "ไม่มีโรคหัวใจ (0)",
        "มีโรคหัวใจ (1)"
    ]

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**จำนวนข้อมูลแต่ละกลุ่ม**")
        st.bar_chart(dt1)

    with right:
        st.markdown("**สัดส่วนข้อมูล**")
        st.write(dt1)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

Knn_model.fit(X_train, y_train)

y_pred = Knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.markdown(
    '<div class="section-title">Model Performance</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Test Data",
            f"{len(X_test)} Records"
        )

    st.markdown("**Confusion Matrix**")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        ),
        use_container_width=True
    )


st.markdown(
    '<div class="section-title">Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.write("กรอกข้อมูลเพื่อทดลองจำแนกผลด้วยโมเดล KNN")


with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("**Personal Information**")

        age = st.slider(
            "อายุ (Age)",
            1, 100, 50
        )

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

        st.markdown("**Blood & Heart Data**")

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

        st.markdown("**Additional Information**")

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

    st.markdown("---")

    predict_button = st.button(
        "วิเคราะห์ข้อมูลด้วย KNN",
        use_container_width=True
    )

    if predict_button:

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

        probability = Knn_model.predict_proba(x_input)[0]

        st.markdown("### Prediction Result")

        if out[0] == 1:
            st.error(
                "ผลการจำแนก: HeartDisease = 1"
            )
            st.write("โมเดลจัดข้อมูลนี้อยู่ในกลุ่ม 1")

        else:
            st.success(
                "ผลการจำแนก: HeartDisease = 0"
            )
            st.write("โมเดลจัดข้อมูลนี้อยู่ในกลุ่ม 0")

        p1, p2 = st.columns(2)

        p1.metric(
            "กลุ่ม 0",
            f"{probability[0] * 100:.2f}%"
        )

        p2.metric(
            "กลุ่ม 1",
            f"{probability[1] * 100:.2f}%"
        )

        st.caption(
            "ความน่าจะเป็นเป็นค่าจากเพื่อนบ้านของโมเดล "
            "ไม่ใช่โอกาสป่วยจริง"
        )


st.markdown("---")

st.markdown(
    '<div class="section-title" style="text-align:center; border:none;">About the Developer</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    left, center, right = st.columns([1, 2, 1])

    with center:

        image_left, profile, image_right = st.columns([1, 2, 1])

        with profile:
            st.image(
                "./img/Profile.jpg",
                use_container_width=True
            )

        st.markdown("""
        <div class="profile-card">
            <h3>ผู้จัดทำโครงงาน</h3>
            <p>สาขาเทคโนโลยีสารสนเทศ</p>
            <p>Heart Disease Prediction using KNN</p>
            <p>2026</p>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div style="
    text-align:center;
    padding:20px;
    color:#9386A9;
    font-size:12px;
">
    HEART DISEASE PREDICTION · MACHINE LEARNING · KNN
    <br>
    ผลลัพธ์ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์
</div>
""", unsafe_allow_html=True)
