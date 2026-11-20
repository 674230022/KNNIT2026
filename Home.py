
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="HeartCare | Prediction",
    page_icon="💜",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Prompt', sans-serif;
}

.stApp {
    background: #171326;
    color: #F4EFFB;
}

.block-container {
    max-width: 1250px;
    padding: 2rem 2rem 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, h4, p, label {
    color: #F4EFFB;
}

.hero {
    background: linear-gradient(115deg, #302344, #211A31);
    border: 1px solid #453653;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 28px;
}

.hero-label {
    color: #BBA0F3;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 600;
}

.hero h1 {
    font-size: 36px;
    font-weight: 600;
    margin: 12px 0 5px;
}

.hero p {
    color: #BDB2CE;
    font-size: 14px;
    margin-bottom: 0;
}

.section-heading {
    color: #D9C7FF;
    font-size: 20px;
    font-weight: 600;
    margin: 32px 0 16px;
}

.section-caption {
    color: #A99DBD;
    font-size: 13px;
    margin-top: -12px;
    margin-bottom: 16px;
}

.metric-card {
    background: #251E35;
    border: 1px solid #40334F;
    padding: 20px;
    border-radius: 14px;
    min-height: 112px;
}

.metric-label {
    color: #AFA1C5;
    font-size: 12px;
}

.metric-value {
    color: #D8C3FF;
    font-size: 27px;
    font-weight: 600;
    margin-top: 8px;
}

.metric-sub {
    color: #9385A9;
    font-size: 11px;
}

.gallery-card {
    background: #241D34;
    border: 1px solid #40334F;
    border-radius: 14px;
    padding: 10px;
    height: 100%;
}

.gallery-title {
    color: #E5D9F6;
    font-size: 14px;
    font-weight: 500;
    margin-top: 12px;
}

.gallery-caption {
    color: #A99DBD;
    font-size: 11px;
}

div[data-testid="stImage"] img {
    border-radius: 9px;
}

div[data-testid="stMetric"] {
    background: #251E35;
    border: 1px solid #40334F;
    border-radius: 14px;
    padding: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #B8A9CE;
}

div[data-testid="stMetricValue"] {
    color: #D8C3FF;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #211A30;
    border: 1px solid #40334F;
    border-radius: 14px;
}

.stButton > button {
    background: #7958B5;
    color: white;
    border: none;
    border-radius: 9px;
    height: 45px;
    font-weight: 500;
}

.stButton > button:hover {
    background: #9472D0;
    color: white;
}

.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
    background: #292239;
    color: #F4EFFB;
    border-color: #514264;
    border-radius: 8px;
}

.stSlider label,
.stSelectbox label,
.stNumberInput label {
    color: #D4C7E5 !important;
}

.stCheckbox label {
    color: #D4C7E5;
}

[data-testid="stDataFrame"] {
    border: 1px solid #443650;
    border-radius: 10px;
}

.profile-panel {
    background: #251E35;
    border: 1px solid #493A5B;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
}

.profile-name {
    color: #E7D9FF;
    font-size: 19px;
    font-weight: 600;
    margin-top: 12px;
}

.profile-detail {
    color: #AFA1C5;
    font-size: 12px;
}

.footer {
    text-align: center;
    color: #847795;
    font-size: 11px;
    padding: 25px 0 5px;
}

hr {
    border-color: #40334F;
}

</style>
""", unsafe_allow_html=True)


dt = pd.read_csv("./Heart3.csv")

X = dt.drop("HeartDisease", axis=1)
y = dt["HeartDisease"]

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


st.markdown("""
<div class="hero">
    <div class="hero-label">HEALTH ANALYTICS · MACHINE LEARNING</div>
    <h1>Heart Disease Prediction</h1>
    <p>
        ระบบจำแนกข้อมูลโรคหัวใจด้วยเทคนิค
        K-Nearest Neighbors (KNN)
    </p>
</div>
""", unsafe_allow_html=True)


m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">TOTAL DATASET</div>
        <div class="metric-value">{len(dt):,}</div>
        <div class="metric-sub">ข้อมูลทั้งหมด</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">FEATURES</div>
        <div class="metric-value">{len(X.columns)}</div>
        <div class="metric-sub">คุณลักษณะข้อมูล</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">ALGORITHM</div>
        <div class="metric-value">KNN</div>
        <div class="metric-sub">K = 5 Neighbors</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown(
    '<div class="section-heading">01 / Heart Health Gallery</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">ความรู้เบื้องต้นเกี่ยวกับระบบหัวใจ</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
    st.image("./img/heart_structure.jpg", use_container_width=True)
    st.markdown('<div class="gallery-title">Heart Anatomy</div>', unsafe_allow_html=True)
    st.markdown('<div class="gallery-caption">โครงสร้างและส่วนประกอบของหัวใจ</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
    st.image("./img/ecg.jpg", use_container_width=True)
    st.markdown('<div class="gallery-title">Electrocardiogram</div>', unsafe_allow_html=True)
    st.markdown('<div class="gallery-caption">คลื่นไฟฟ้าหัวใจ (ECG)</div></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
    st.image("./img/heart_health.jpg", use_container_width=True)
    st.markdown('<div class="gallery-title">Heart Wellness</div>', unsafe_allow_html=True)
    st.markdown('<div class="gallery-caption">การดูแลสุขภาพหัวใจ</div></div>', unsafe_allow_html=True)


st.markdown(
    '<div class="section-heading">02 / Dataset Overview</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown("**ข้อมูลตัวอย่างจาก Heart3.csv**")

    styled_data = (
        dt.head(10).style
        .set_properties(**{
            "background-color": "#251E35",
            "color": "#E9E0F5",
            "border-color": "#40334F"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#382B4D"),
                    ("color", "#D9C7FF"),
                    ("border-color", "#514264"),
                    ("text-align", "left")
                ]
            }
        ])
    )

    st.dataframe(
        styled_data,
        use_container_width=True,
        height=350
    )


st.markdown(
    '<div class="section-heading">03 / Heart Disease Statistics</div>',
    unsafe_allow_html=True
)

counts = dt["HeartDisease"].value_counts().reindex([0, 1], fill_value=0)

chart_data = pd.DataFrame({
    "ผลการจำแนก": ["ไม่มีโรคหัวใจ", "มีโรคหัวใจ"],
    "จำนวน": [int(counts.loc[0]), int(counts.loc[1])]
})

c1, c2 = st.columns([1.35, 1])

with c1:
    with st.container(border=True):
        st.markdown("**จำนวนข้อมูลแต่ละกลุ่ม**")

        fig = px.bar(
            chart_data,
            x="ผลการจำแนก",
            y="จำนวน",
            color="ผลการจำแนก",
            color_discrete_sequence=["#9275C9", "#C09AF5"],
            text="จำนวน"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E9E0F5",
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="",
            yaxis_title="จำนวนรายการ"
        )

        fig.update_traces(
            textposition="outside",
            marker_line_width=0
        )

        st.plotly_chart(fig, use_container_width=True)

with c2:
    with st.container(border=True):
        st.markdown("**สัดส่วนข้อมูล**")

        pie = go.Figure(data=[go.Pie(
            labels=["ไม่มีโรคหัวใจ", "มีโรคหัวใจ"],
            values=[int(counts.loc[0]), int(counts.loc[1])],
            hole=0.68,
            marker=dict(
                colors=["#9275C9", "#C09AF5"],
                line=dict(color="#211A30", width=3)
            ),
            textinfo="percent",
            textfont=dict(color="#FFFFFF", size=13)
        )])

        pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E9E0F5",
            showlegend=True,
            legend=dict(orientation="h", y=-0.12),
            margin=dict(l=5, r=5, t=15, b=25)
        )

        st.plotly_chart(pie, use_container_width=True)


st.markdown(
    '<div class="section-heading">04 / Model Performance</div>',
    unsafe_allow_html=True
)

p1, p2, p3 = st.columns(3)

with p1:
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

with p2:
    st.metric("Training Data", f"{len(X_train):,}")

with p3:
    st.metric("Testing Data", f"{len(X_test):,}")


with st.container(border=True):
    st.markdown("**Confusion Matrix**")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    cm_fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        color_continuous_scale=[
            "#211A30",
            "#7657A8",
            "#C09AF5"
        ],
        aspect="auto"
    )

    cm_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E9E0F5",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=15, b=10)
    )

    st.plotly_chart(cm_fig, use_container_width=True)


st.markdown(
    '<div class="section-heading">05 / Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-caption">กรอกข้อมูลเพื่อทดลองจำแนกผลด้วยโมเดล KNN</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown("**Personal Information**")

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

    st.markdown("---")

    predict_button = st.button(
        "วิเคราะห์ข้อมูล",
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
            st.error("ผลการจำแนก: HeartDisease = 1")
        else:
            st.success("ผลการจำแนก: HeartDisease = 0")

        result1, result2 = st.columns(2)

        result1.metric(
            "กลุ่ม 0",
            f"{probability[0] * 100:.2f}%"
        )

        result2.metric(
            "กลุ่ม 1",
            f"{probability[1] * 100:.2f}%"
        )

        st.caption(
            "ผลลัพธ์นี้เป็นการทำนายจากแบบจำลองเพื่อการศึกษา "
            "ไม่ใช่การวินิจฉัยทางการแพทย์"
        )


st.markdown(
    '<div class="section-heading" style="text-align:center;">About the Developer</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.image(
            "./img/Profile.jpg",
            use_container_width=True
        )

        st.markdown("""
        <div class="profile-panel">
            <div class="profile-name">ผู้จัดทำโครงงาน</div>
            <div class="profile-detail">
                สาขาเทคโนโลยีสารสนเทศ
                <br>
                Heart Disease Prediction using KNN
                <br>
                2026
            </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    HEARTCARE · MACHINE LEARNING · K-NEAREST NEIGHBORS
    <br>
    FOR EDUCATIONAL PURPOSES ONLY
</div>
""", unsafe_allow_html=True)
