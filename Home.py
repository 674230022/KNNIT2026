from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="HeartCare | Heart Disease Management",
    page_icon="💜",
    layout="wide"
)

DATA_PATH = "Heart3.csv"

FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]

TARGET = "HeartDisease"
COLUMNS = FEATURES + [TARGET]


# =========================================================
# CUSTOM CSS DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

:root {
    --page: #100d1b;
    --surface: #191426;
    --surface-2: #211a31;
    --surface-3: #2a2140;
    --line: #382d4c;
    --text: #f5f1fc;
    --muted: #b7abc9;
    --purple: #a78bfa;
    --purple-strong: #8b5cf6;
}

html, body, [class*="css"], button, input, textarea {
    font-family: 'Prompt', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(124, 58, 237, .16), transparent 34%),
        radial-gradient(ellipse at 90% 8%, rgba(168, 85, 247, .10), transparent 28%),
        var(--page);
    color: var(--text);
}

.block-container {
    max-width: 1320px;
    padding: 2.2rem 2.4rem 3.5rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

h1, h2, h3, h4, p, label, span {
    font-family: 'Prompt', sans-serif;
}

h1, h2, h3, h4 {
    color: var(--text);
    letter-spacing: -.3px;
}

p, label {
    color: var(--muted);
}

/* Main heading */

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        115deg,
        #29203d 0%,
        #211a33 54%,
        #1a1729 100%
    );
    border: 1px solid #49375f;
    border-radius: 22px;
    padding: 34px 38px;
    margin: 0 0 26px;
    box-shadow: 0 18px 48px rgba(0,0,0,.22);
}

.hero:after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    border-radius: 50%;
    right: -65px;
    top: -105px;
    background: radial-gradient(
        circle,
        rgba(167,139,250,.28),
        rgba(167,139,250,0)
    );
    pointer-events: none;
}

.hero small {
    color: #c4a9ff;
    letter-spacing: 2.3px;
    font-size: 11px;
    font-weight: 600;
}

.hero h1 {
    font-size: clamp(28px, 3vw, 38px);
    line-height: 1.25;
    margin: 12px 0 8px;
    color: #fff;
    font-weight: 700;
}

.hero p {
    color: #c9bfd8;
    margin: 0;
    font-size: 14px;
}

/* Section headings and metric cards */

.section {
    color: #e1d3ff;
    font-size: 19px;
    font-weight: 600;
    margin: 30px 0 15px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #211a31, #1a1627);
    border: 1px solid var(--line);
    border-radius: 17px;
    padding: 19px 21px;
    box-shadow: 0 8px 22px rgba(0,0,0,.12);
}

div[data-testid="stMetricLabel"] {
    color: #b9accd;
    font-size: 13px;
}

div[data-testid="stMetricValue"] {
    color: #d8c5ff;
    font-weight: 700;
    font-size: 30px;
}

/* Streamlit containers and gallery cards */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(29, 23, 43, .88);
    border: 1px solid var(--line);
    border-radius: 17px;
    padding: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,.10);
}

div[data-testid="stImage"] img {
    border-radius: 12px;
    object-fit: cover;
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: 7px;
    border-bottom: 1px solid var(--line);
    padding: 0 0 8px;
}

.stTabs [data-baseweb="tab"] {
    background: #1b1628;
    color: #c5b8d8;
    border: 1px solid #342943;
    border-radius: 10px 10px 8px 8px;
    padding: 10px 15px;
    height: auto;
}

.stTabs [aria-selected="true"] {
    background: #392957 !important;
    color: #f2eaff !important;
    border-color: #7656a8 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: var(--purple);
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #8b5cf6, #7045c4);
    color: #fff;
    border: 1px solid #9b79ed;
    border-radius: 11px;
    min-height: 43px;
    padding: 8px 18px;
    font-weight: 600;
    transition: all .18s ease;
    box-shadow: 0 5px 14px rgba(112,69,196,.18);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #9b72fa, #8058d6);
    border-color: #b59aff;
    color: #fff;
    transform: translateY(-1px);
}

.stButton > button:disabled {
    background: #40364f;
    color: #9e93ad;
    border-color: #4a3d5b;
    box-shadow: none;
}

/* Inputs */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stTextArea textarea {
    background: #211a30 !important;
    color: #f4effb !important;
    border: 1px solid #493b5d !important;
    border-radius: 10px !important;
}

.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 1px #a78bfa !important;
}

[data-baseweb="popover"] {
    background: #211a30;
    border: 1px solid var(--line);
}

[data-baseweb="menu"] {
    background: #211a30;
}

[data-baseweb="menu"] li {
    color: var(--text);
}

/* Tables, charts and messages */

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 13px;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: var(--line);
}

.note {
    color: #a99dbd;
    font-size: 12px;
}

/* Responsive design */

@media (max-width: 800px) {
    .block-container {
        padding: 1.1rem 1rem 2.5rem;
    }

    .hero {
        padding: 25px 22px;
        border-radius: 17px;
    }

    .hero h1 {
        font-size: 27px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 8px 10px;
        font-size: 12px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA MANAGEMENT
# =========================================================

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"ไม่พบไฟล์ {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    missing = [c for c in COLUMNS if c not in data.columns]

    if missing:
        raise ValueError("คอลัมน์ไม่ครบ: " + ", ".join(missing))

    data = data[COLUMNS].copy()

    for c in COLUMNS:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna().reset_index(drop=True)

    return data


def save_data(data):
    data[COLUMNS].to_csv(
        DATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    st.cache_data.clear()


# =========================================================
# KNN MODEL
# =========================================================

def train_model(data):

    if data[TARGET].nunique() < 2:
        raise ValueError(
            "ต้องมีข้อมูลทั้งกลุ่ม 0 และ 1 จึงจะฝึกโมเดลได้"
        )

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    counts = y.value_counts()

    if counts.min() < 2:
        raise ValueError(
            "แต่ละกลุ่มต้องมีอย่างน้อย 2 รายการเพื่อแบ่ง Train/Test"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    k = min(5, len(X_train))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return (
        model,
        accuracy_score(y_test, pred),
        confusion_matrix(y_test, pred, labels=[0, 1]),
        len(X_train),
        len(X_test),
        k
    )


# =========================================================
# LOAD DATA
# =========================================================

try:
    dt = load_data()

except Exception as e:
    st.error(f"อ่านข้อมูลไม่สำเร็จ: {e}")
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <small>HEALTH ANALYTICS · MACHINE LEARNING</small>
    <h1>HeartCare Management</h1>
    <p>
        ระบบจัดการข้อมูลและจำแนกโรคหัวใจด้วย
        K-Nearest Neighbors
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD METRICS
# =========================================================

m1, m2, m3 = st.columns(3)

m1.metric(
    "ข้อมูลทั้งหมด",
    f"{len(dt):,} รายการ"
)

m2.metric(
    "คุณลักษณะ",
    len(FEATURES)
)

m3.metric(
    "จำนวนกลุ่ม",
    int(dt[TARGET].nunique())
)


# =========================================================
# HEART HEALTH GALLERY
# =========================================================

st.markdown(
    '<div class="section">Heart Health Gallery</div>',
    unsafe_allow_html=True
)

gallery = st.columns(3)

gallery_items = [
    (
        "img/heart_structure.jpg",
        "Heart Anatomy",
        "โครงสร้างหัวใจ"
    ),
    (
        "img/ecg.jpg",
        "Electrocardiogram",
        "คลื่นไฟฟ้าหัวใจ"
    ),
    (
        "img/heart_health.jpg",
        "Heart Wellness",
        "การดูแลสุขภาพหัวใจ"
    )
]

for col, (path, title, desc) in zip(gallery, gallery_items):

    with col:

        with st.container(border=True):

            if os.path.exists(path):
                st.image(
                    path,
                    use_container_width=True
                )

            st.markdown(f"**{title}**")
            st.caption(desc)


# =========================================================
# CRUD TABS
# =========================================================

st.markdown(
    '<div class="section">จัดการข้อมูล Heart3.csv</div>',
    unsafe_allow_html=True
)

tab_dash, tab_add, tab_edit, tab_delete, tab_search, tab_model = st.tabs([
    "ภาพรวม",
    "เพิ่มข้อมูล",
    "แก้ไขข้อมูล",
    "ลบข้อมูล",
    "ค้นหา",
    "KNN Model"
])


# =========================================================
# TAB 1: DASHBOARD
# =========================================================

with tab_dash:

    st.markdown("**ข้อมูลตัวอย่าง 10 รายการแรก**")

    st.dataframe(
        dt.head(10),
        use_container_width=True,
        hide_index=True
    )

    counts = dt[TARGET].value_counts().reindex(
        [0, 1],
        fill_value=0
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.markdown("**จำนวนข้อมูลในแต่ละกลุ่ม**")

        st.bar_chart(
            pd.DataFrame(
                {
                    "จำนวนรายการ": [
                        int(counts.loc[0]),
                        int(counts.loc[1])
                    ]
                },
                index=[
                    "ไม่มีโรคหัวใจ (0)",
                    "มีโรคหัวใจ (1)"
                ]
            )
        )

    with chart_col2:

        st.markdown("**สถิติข้อมูลเบื้องต้น**")

        st.dataframe(
            dt[FEATURES].describe().round(2),
            use_container_width=True
        )


# =========================================================
# TAB 2: ADD DATA
# =========================================================

with tab_add:

    st.markdown("**เพิ่มข้อมูลใหม่**")

    st.caption(
        "กรอกข้อมูลตามรูปแบบตัวเลขและรหัสที่มีอยู่ใน Heart3.csv"
    )

    with st.form("add_record_form"):

        a, b, c = st.columns(3)

        with a:

            age = st.number_input(
                "Age",
                1,
                120,
                50
            )

            sex = st.selectbox(
                "Sex",
                sorted(dt.Sex.unique().tolist())
            )

            chest = st.selectbox(
                "ChestPainType",
                sorted(dt.ChestPainType.unique().tolist())
            )

            restingbp = st.number_input(
                "RestingBP",
                0,
                300,
                120
            )

        with b:

            cholesterol = st.number_input(
                "Cholesterol",
                0,
                1000,
                200
            )

            fastingbs = st.selectbox(
                "FastingBS",
                sorted(dt.FastingBS.unique().tolist())
            )

            restingecg = st.selectbox(
                "RestingECG",
                sorted(dt.RestingECG.unique().tolist())
            )

            maxhr = st.number_input(
                "MaxHR",
                1,
                300,
                150
            )

        with c:

            exercise = st.selectbox(
                "ExerciseAngina",
                sorted(dt.ExerciseAngina.unique().tolist())
            )

            oldpeak = st.number_input(
                "Oldpeak",
                -10.0,
                20.0,
                0.0,
                step=0.1
            )

            slope = st.selectbox(
                "ST_Slope",
                sorted(dt.ST_Slope.unique().tolist())
            )

            target = st.selectbox(
                "HeartDisease (ผลลัพธ์)",
                [0, 1],
                format_func=lambda x:
                    "0 - ไม่มีโรคหัวใจ"
                    if x == 0
                    else "1 - มีโรคหัวใจ"
            )

        submitted = st.form_submit_button(
            "บันทึกข้อมูลใหม่",
            use_container_width=True
        )

        if submitted:

            new_row = {
                "Age": age,
                "Sex": sex,
                "ChestPainType": chest,
                "RestingBP": restingbp,
                "Cholesterol": cholesterol,
                "FastingBS": fastingbs,
                "RestingECG": restingecg,
                "MaxHR": maxhr,
                "ExerciseAngina": exercise,
                "Oldpeak": oldpeak,
                "ST_Slope": slope,
                "HeartDisease": target
            }

            updated = pd.concat(
                [dt, pd.DataFrame([new_row])],
                ignore_index=True
            )

            save_data(updated)

            st.success(
                "เพิ่มข้อมูลและบันทึกลง Heart3.csv แล้ว"
            )

            st.rerun()


# =========================================================
# TAB 3: EDIT DATA
# =========================================================

with tab_edit:

    st.markdown("**แก้ไขรายการข้อมูล**")

    if len(dt):

        row_no = st.selectbox(
            "เลือกรายการที่ต้องการแก้ไข (หมายเลขแถว)",
            list(range(len(dt))),
            format_func=lambda i:
                f"รายการ {i+1} | อายุ {int(dt.iloc[i]['Age'])} "
                f"| ผล {int(dt.iloc[i][TARGET])}"
        )

        current = dt.iloc[row_no]

        with st.form("edit_record_form"):

            a, b, c = st.columns(3)

            with a:

                e_age = st.number_input(
                    "Age",
                    1,
                    120,
                    int(current.Age),
                    key="e_age"
                )

                e_sex = st.selectbox(
                    "Sex",
                    sorted(set(dt.Sex.tolist())),
                    index=sorted(dt.Sex.unique().tolist()).index(
                        current.Sex
                    ),
                    key="e_sex"
                )

                e_chest = st.selectbox(
                    "ChestPainType",
                    sorted(dt.ChestPainType.unique().tolist()),
                    index=sorted(
                        dt.ChestPainType.unique().tolist()
                    ).index(current.ChestPainType),
                    key="e_chest"
                )

                e_bp = st.number_input(
                    "RestingBP",
                    0,
                    300,
                    int(current.RestingBP),
                    key="e_bp"
                )

            with b:

                e_chol = st.number_input(
                    "Cholesterol",
                    0,
                    1000,
                    int(current.Cholesterol),
                    key="e_chol"
                )

                e_fbs = st.selectbox(
                    "FastingBS",
                    sorted(dt.FastingBS.unique().tolist()),
                    index=sorted(
                        dt.FastingBS.unique().tolist()
                    ).index(current.FastingBS),
                    key="e_fbs"
                )

                e_ecg = st.selectbox(
                    "RestingECG",
                    sorted(dt.RestingECG.unique().tolist()),
                    index=sorted(
                        dt.RestingECG.unique().tolist()
                    ).index(current.RestingECG),
                    key="e_ecg"
                )

                e_maxhr = st.number_input(
                    "MaxHR",
                    1,
                    300,
                    int(current.MaxHR),
                    key="e_maxhr"
                )

            with c:

                e_ex = st.selectbox(
                    "ExerciseAngina",
                    sorted(dt.ExerciseAngina.unique().tolist()),
                    index=sorted(
                        dt.ExerciseAngina.unique().tolist()
                    ).index(current.ExerciseAngina),
                    key="e_ex"
                )

                e_old = st.number_input(
                    "Oldpeak",
                    -10.0,
                    20.0,
                    float(current.Oldpeak),
                    step=0.1,
                    key="e_old"
                )

                e_slope = st.selectbox(
                    "ST_Slope",
                    sorted(dt.ST_Slope.unique().tolist()),
                    index=sorted(
                        dt.ST_Slope.unique().tolist()
                    ).index(current.ST_Slope),
                    key="e_slope"
                )

                e_target = st.selectbox(
                    "HeartDisease",
                    [0, 1],
                    index=int(current.HeartDisease),
                    key="e_target"
                )

            edit_submit = st.form_submit_button(
                "บันทึกการแก้ไข",
                use_container_width=True
            )

            if edit_submit:

                dt.loc[row_no, COLUMNS] = [
                    e_age,
                    e_sex,
                    e_chest,
                    e_bp,
                    e_chol,
                    e_fbs,
                    e_ecg,
                    e_maxhr,
                    e_ex,
                    e_old,
                    e_slope,
                    e_target
                ]

                save_data(dt)

                st.success(
                    "แก้ไขข้อมูลเรียบร้อยแล้ว"
                )

                st.rerun()

    else:

        st.info("ยังไม่มีข้อมูล")


# =========================================================
# TAB 4: DELETE DATA
# =========================================================

with tab_delete:

    st.markdown("**ลบรายการข้อมูล**")

    st.warning(
        "การลบจะเปลี่ยนแปลงไฟล์ Heart3.csv โดยตรง"
    )

    if len(dt):

        del_no = st.selectbox(
            "เลือกรายการที่ต้องการลบ",
            list(range(len(dt))),
            format_func=lambda i:
                f"รายการ {i+1} | อายุ {int(dt.iloc[i]['Age'])} "
                f"| ผล {int(dt.iloc[i][TARGET])}"
        )

        st.dataframe(
            dt.iloc[[del_no]],
            use_container_width=True,
            hide_index=True
        )

        confirm = st.checkbox(
            "ฉันยืนยันว่าต้องการลบรายการนี้"
        )

        if st.button(
            "ลบข้อมูลที่เลือก",
            disabled=not confirm,
            use_container_width=True
        ):

            updated = dt.drop(
                index=del_no
            ).reset_index(drop=True)

            save_data(updated)

            st.success(
                "ลบข้อมูลเรียบร้อยแล้ว"
            )

            st.rerun()

    else:

        st.info("ไม่มีข้อมูลให้ลบ")


# =========================================================
# TAB 5: SEARCH DATA
# =========================================================

with tab_search:

    st.markdown("**ค้นหาข้อมูลใน Dataset**")

    search_col, search_val = st.columns([1, 2])

    with search_col:

        search_field = st.selectbox(
            "ค้นหาจากคอลัมน์",
            COLUMNS
        )

    with search_val:

        search_text = st.text_input(
            "พิมพ์ค่าที่ต้องการค้นหา"
        )

    if search_text.strip():

        result = dt[
            dt[search_field]
            .astype(str)
            .str.contains(
                search_text.strip(),
                case=False,
                na=False
            )
        ]

        st.write(
            f"พบข้อมูล {len(result)} รายการ"
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.caption(
            "ตัวอย่าง: ค้นหาอายุ 50 หรือ HeartDisease 1"
        )


# =========================================================
# TAB 6: KNN MODEL
# =========================================================

with tab_model:

    st.markdown("**ฝึกโมเดลใหม่จากข้อมูลปัจจุบัน**")

    st.write(
        f"ข้อมูลที่ใช้ฝึก: {len(dt):,} รายการ"
    )

    if st.button(
        "ฝึก KNN ใหม่",
        use_container_width=True
    ):

        try:

            model, acc, cm, train_n, test_n, k = train_model(dt)

            st.session_state["knn_model"] = model
            st.session_state["knn_accuracy"] = acc
            st.session_state["knn_cm"] = cm
            st.session_state["knn_train_n"] = train_n
            st.session_state["knn_test_n"] = test_n
            st.session_state["knn_k"] = k

            st.success(
                "ฝึกโมเดลใหม่สำเร็จ โดยใช้ข้อมูลล่าสุด"
            )

        except Exception as e:

            st.error(
                f"ไม่สามารถฝึกโมเดลได้: {e}"
            )

    try:

        model, acc, cm, train_n, test_n, k = train_model(dt)

        q1, q2, q3 = st.columns(3)

        q1.metric(
            "Accuracy",
            f"{acc*100:.2f}%"
        )

        q2.metric(
            "Training",
            f"{train_n:,}"
        )

        q3.metric(
            "Testing",
            f"{test_n:,}"
        )

        st.caption(
            f"ใช้ K = {k} | โมเดลจะถูกฝึกใหม่อัตโนมัติเมื่อมีการเปลี่ยนข้อมูล"
        )

        st.markdown("**Confusion Matrix**")

        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"]
            ),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"สถานะโมเดล: {e}"
        )


# =========================================================
# PREDICTION
# =========================================================

st.markdown(
    '<div class="section">ทำนายข้อมูลใหม่ด้วย KNN</div>',
    unsafe_allow_html=True
)

try:

    model, acc, cm, train_n, test_n, k = train_model(dt)

    with st.container(border=True):

        a, b, c = st.columns(3)

        with a:

            p_age = st.number_input(
                "อายุ (Age)",
                1,
                120,
                50,
                key="p_age"
            )

            p_sex = st.selectbox(
                "Sex",
                sorted(dt.Sex.unique().tolist()),
                key="p_sex"
            )

            p_chest = st.selectbox(
                "ChestPainType",
                sorted(dt.ChestPainType.unique().tolist()),
                key="p_chest"
            )

            p_bp = st.number_input(
                "RestingBP",
                0,
                300,
                120,
                key="p_bp"
            )

        with b:

            p_chol = st.number_input(
                "Cholesterol",
                0,
                1000,
                200,
                key="p_chol"
            )

            p_fbs = st.selectbox(
                "FastingBS",
                sorted(dt.FastingBS.unique().tolist()),
                key="p_fbs"
            )

            p_ecg = st.selectbox(
                "RestingECG",
                sorted(dt.RestingECG.unique().tolist()),
                key="p_ecg"
            )

            p_maxhr = st.number_input(
                "MaxHR",
                1,
                300,
                150,
                key="p_maxhr"
            )

        with c:

            p_ex = st.selectbox(
                "ExerciseAngina",
                sorted(dt.ExerciseAngina.unique().tolist()),
                key="p_ex"
            )

            p_old = st.number_input(
                "Oldpeak",
                -10.0,
                20.0,
                0.0,
                step=0.1,
                key="p_old"
            )

            p_slope = st.selectbox(
                "ST_Slope",
                sorted(dt.ST_Slope.unique().tolist()),
                key="p_slope"
            )

        if st.button(
            "วิเคราะห์ข้อมูล",
            use_container_width=True
        ):

            new_data = pd.DataFrame(
                [[
                    p_age,
                    p_sex,
                    p_chest,
                    p_bp,
                    p_chol,
                    p_fbs,
                    p_ecg,
                    p_maxhr,
                    p_ex,
                    p_old,
                    p_slope
                ]],
                columns=FEATURES
            )

            result = int(
                model.predict(new_data)[0]
            )

            probs = model.predict_proba(new_data)[0]

            if result == 1:

                st.error(
                    "ผลการจำแนก: HeartDisease = 1"
                )

            else:

                st.success(
                    "ผลการจำแนก: HeartDisease = 0"
                )

            r1, r2 = st.columns(2)

            r1.metric(
                "กลุ่ม 0",
                f"{probs[list(model.classes_).index(0)]*100:.2f}%"
            )

            r2.metric(
                "กลุ่ม 1",
                f"{probs[list(model.classes_).index(1)]*100:.2f}%"
            )

            st.caption(
                "ผลลัพธ์เพื่อการศึกษา ไม่ใช่การวินิจฉัยทางการแพทย์"
            )

except Exception as e:

    st.warning(
        f"ยังไม่สามารถทำนายได้: {e}"
    )


# =========================================================
# PROFILE
# =========================================================

st.markdown("---")

profile_path = "img/Profile.jpg"

if os.path.exists(profile_path):

    left, center, right = st.columns([1, 1, 1])

    with center:

        st.image(
            profile_path,
            use_container_width=True
        )

        st.markdown(
            """
            <center>
                <b>ผู้จัดทำโครงงาน</b>
                <br>
                <small>สาขาเทคโนโลยีสารสนเทศ</small>
            </center>
            """,
            unsafe_allow_html=True
        )

else:

    st.markdown(
        """
        <center>
            <b>ผู้จัดทำโครงงาน</b>
            <br>
            <small>สาขาเทคโนโลยีสารสนเทศ</small>
        </center>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div style="
    text-align:center;
    color:#847795;
    font-size:11px;
    padding:22px 0;
">
    HEARTCARE · MACHINE LEARNING · K-NEAREST NEIGHBORS
    <br>
    ผลลัพธ์ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์
</div>
""", unsafe_allow_html=True)