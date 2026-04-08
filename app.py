import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# =========================
# ページ設定
# =========================
st.set_page_config(
    page_title="小児腹部鈍的外傷 意思決定支援ツール",
    layout="wide"
)

# =========================
# 見た目
# =========================
st.markdown("""
<style>
.big-font {
    font-size:22px !important;
    font-weight: bold;
}
.low-risk {
    color: green;
    font-size: 28px;
    font-weight: bold;
}
.mid-risk {
    color: orange;
    font-size: 28px;
    font-weight: bold;
}
.high-risk {
    color: red;
    font-size: 28px;
    font-weight: bold;
}
.block {
    padding: 1rem;
    border-radius: 0.75rem;
    background-color: #f7f7f7;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 定数
# =========================
LAB_COLS = ["AST", "ALT", "Amylase", "Lipase", "WBC", "Hematocrit", "BUN", "Creatinine", "HCO3"]

PECARN_FIELDS = [
    ("AbdomenPain", "腹痛"),
    ("VomitWretch", "嘔吐"),
    ("DecrBreathSound", "呼吸音減弱"),
    ("AbdTrauma", "腹部外傷"),
    ("AbdomenTender", "腹部圧痛"),
    ("ThoracicTrauma", "胸部外傷"),
    ("SeatBeltSign", "シートベルトサイン"),
]

LAB_FIELDS = [
    ("AST", "AST"),
    ("ALT", "ALT"),
    ("Amylase", "Amylase"),
    ("Lipase", "Lipase"),
    ("WBC", "WBC"),
    ("Hematocrit", "Hematocrit"),
    ("BUN", "BUN"),
    ("Creatinine", "Creatinine"),
    ("HCO3", "HCO3"),
]

# =========================
# 関数
# =========================
def symptom_to_code(label: str) -> int:
    return {"はい": 1, "いいえ": 2, "不明": 3}[label]

def to_float_or_nan(x):
    try:
        if str(x).strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def compute_acs(row: pd.Series) -> int:
    return int(
        (row["AbdomenPain"] == 1) +
        (row["VomitWretch"] == 1) +
        (row["DecrBreathSound"] == 1) +
        (row["AbdTrauma"] == 1) +
        (row["AbdomenTender"] == 1) +
        (row["ThoracicTrauma"] == 1) +
        (row["SeatBeltSign"] == 1) +
        (row["GCSScore"] < 14)
    )

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    # 初期臨床情報
    X["AgeYears"] = df["AgeYears"]
    X["Sex_male"] = df["Sex_male"]
    X["SystolicBP"] = df["SystolicBP"]
    X["Pulse"] = df["Pulse"]
    X["RespRate"] = df["RespRate"]
    X["ShockIndex"] = df["ShockIndex"]

    # ACS
    X["ACS"] = df["ACS"]

    # PECARN個別項目
    X["GCS_lt14"] = (df["GCSScore"] < 14).astype(int)

    for col, _ in PECARN_FIELDS:
        X[f"{col}_yes"] = (df[col] == 1).astype(int)
        X[f"{col}_unk"] = df[col].isin([3, 4]).astype(int)

    # 採血
    for col in LAB_COLS:
        X[f"{col}_val"] = df[col]
        X[f"{col}_missing"] = df[col].isna().astype(int)

    return X

def classify(prob: float):
    if prob < 0.01:
        return "低", "low-risk", "IAIAI低リスクと考えられます。臨床経過と併せてご判断ください。"
    elif prob < 0.05:
        return "中", "mid-risk", "中間リスクです。身体所見・経過・採血所見を踏まえて追加評価をご検討ください。"
    else:
        return "高", "high-risk", "IAIAI高リスクが示唆されます。CTを含めた追加評価を積極的にご検討ください。"

def final_message(risk: str) -> str:
    if risk == "低":
        return "CTは現時点で回避候補です"
    elif risk == "中":
        return "CT適応は追加評価を踏まえて判断してください"
    else:
        return "CTを積極的に検討してください"

def explain_prediction(df_input: pd.DataFrame, prob: float):
    row = df_input.iloc[0]

    positive_pecarn = []
    if row["GCSScore"] < 14:
        positive_pecarn.append("GCS<14")
    for key, label in PECARN_FIELDS:
        if row[key] == 1:
            positive_pecarn.append(label)

    unknown_pecarn = []
    for key, label in PECARN_FIELDS:
        if row[key] in [3, 4]:
            unknown_pecarn.append(label)

    missing_labs = []
    for lab in LAB_COLS:
        if pd.isna(row[lab]):
            missing_labs.append(lab)

    comments = []

    if row["ACS"] >= 4:
        comments.append("ACSが高く、高リスク群に該当します。")
    elif row["ACS"] == 3:
        comments.append("ACS=3でリスク上昇が示唆されます。")
    else:
        comments.append("ACSは0〜2であり、中間〜低リスク帯です。")

    if row["GCSScore"] < 14:
        comments.append("GCS低下はPECARNの重要所見です。")

    if pd.notna(row["ShockIndex"]) and row["ShockIndex"] >= 1.0:
        comments.append("Shock Indexが高く、循環動態への注意が必要です。")

    if len(positive_pecarn) == 0:
        comments.append("PECARN陽性項目は認めません。")

    if prob < 0.01:
        comments.append("モデル上は低リスクです。")
    elif prob < 0.05:
        comments.append("モデル上は中間リスクです。")
    else:
        comments.append("モデル上は高リスクです。")

    return {
        "positive_pecarn": positive_pecarn,
        "unknown_pecarn": unknown_pecarn,
        "missing_labs": missing_labs,
        "comments": comments,
    }

# =========================
# モデル読み込み
# =========================
@st.cache_resource
def load_bundle():
    model_path = Path("model_c_bundle.joblib")
    if not model_path.exists():
        st.error("model_c_bundle.joblib が見つかりません。先にモデル保存を行ってください。")
        st.stop()
    return joblib.load(model_path)

bundle = load_bundle()

# =========================
# タイトル
# =========================
st.title("小児腹部鈍的外傷 意思決定支援ツール")
st.markdown("#### Model C（ACS + PECARN個別 + 初期臨床情報 + 採血）")
st.caption("IAIAI確率を補助的に表示し、CT適応判断の参考情報を提供します。")
st.markdown("---")

# =========================
# 入力
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("基本情報・バイタル")
    age = st.number_input("年齢", min_value=0.0, max_value=18.0, value=10.0, step=0.1)
    sex = st.selectbox("性別", ["男性", "女性"])
    sbp = st.number_input("収縮期血圧", min_value=20.0, max_value=250.0, value=110.0, step=1.0)
    pulse = st.number_input("脈拍", min_value=20.0, max_value=250.0, value=90.0, step=1.0)
    rr = st.number_input("呼吸数", min_value=5.0, max_value=80.0, value=20.0, step=1.0)

    shock = pulse / sbp if sbp > 0 else np.nan
    st.info(f"Shock Index（自動計算）: {shock:.2f}")

with col2:
    st.subheader("PECARN項目")
    gcs = st.number_input("GCS", min_value=3, max_value=15, value=15, step=1)

    pecarn_inputs = {}
    for key, label in PECARN_FIELDS:
        pecarn_inputs[key] = st.selectbox(label, ["はい", "いいえ", "不明"], index=1)

st.markdown("---")
st.subheader("採血（未測定なら空欄のまま）")

lab_cols = st.columns(3)
lab_inputs = {}

for idx, (key, label) in enumerate(LAB_FIELDS):
    with lab_cols[idx % 3]:
        lab_inputs[key] = st.text_input(label)

# =========================
# 判定
# =========================
if st.button("判定する", use_container_width=True):
    row_data = {
        "AgeYears": age,
        "Sex_male": 1 if sex == "男性" else 0,
        "GCSScore": gcs,
        "SystolicBP": sbp,
        "Pulse": pulse,
        "RespRate": rr,
        "ShockIndex": shock,
    }

    for key, _ in PECARN_FIELDS:
        row_data[key] = symptom_to_code(pecarn_inputs[key])

    for key, _ in LAB_FIELDS:
        row_data[key] = to_float_or_nan(lab_inputs[key])

    df_input = pd.DataFrame([row_data])
    df_input["ACS"] = df_input.apply(compute_acs, axis=1)

    X = build_features(df_input)

    for col in bundle["feature_columns"]:
        if col not in X.columns:
            X[col] = 0

    X = X[bundle["feature_columns"]].copy()

    X[bundle["cols_to_impute"]] = bundle["imputer"].transform(X[bundle["cols_to_impute"]])
    X[bundle["cols_to_scale"]] = bundle["scaler"].transform(X[bundle["cols_to_scale"]])

    prob = bundle["model"].predict_proba(X)[:, 1][0]
    risk, css_class, comment = classify(prob)
    explanation = explain_prediction(df_input, prob)
    main_msg = final_message(risk)

    st.markdown("---")
    st.subheader("最終判定")

    if risk == "低":
        st.success(main_msg)
    elif risk == "中":
        st.warning(main_msg)
    else:
        st.error(main_msg)

    c1, c2, c3 = st.columns(3)
    c1.metric("ACS", int(df_input["ACS"].iloc[0]))
    c2.metric("IAIAI確率", f"{prob*100:.2f}%")
    c3.metric("リスク分類", f"{risk}リスク")

    st.markdown(f"<p class='{css_class}'>{risk}リスク</p>", unsafe_allow_html=True)
    st.info(comment)

    st.markdown("### 判定の根拠")
    st.write(f"**PECARN陽性項目**：{', '.join(explanation['positive_pecarn']) if explanation['positive_pecarn'] else 'なし'}")
    st.write(f"**PECARN不明項目**：{', '.join(explanation['unknown_pecarn']) if explanation['unknown_pecarn'] else 'なし'}")
    st.write(f"**未測定の採血**：{', '.join(explanation['missing_labs']) if explanation['missing_labs'] else 'なし'}")

    st.markdown("**コメント**")
    for txt in explanation["comments"]:
        st.write(f"- {txt}")

    with st.expander("入力内容の確認"):
        st.dataframe(df_input.T, use_container_width=True)

    st.caption("※ 本ツールは臨床判断を補助するものであり、最終判断は担当医が行ってください。")
