# code/code.py
# 실행: streamlit run code/code.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="베이스라인 비교", page_icon="📊", layout="wide")

# ------------------------------------------------------------------
# 0. 고정 설정 (모든 모델이 똑같이 사용)
# ------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
FEATURES = {
    "TotalSteps": "총 걸음 수",
    "VeryActiveMinutes": "고강도 활동(분)",
    "FairlyActiveMinutes": "중강도 활동(분)",
    "LightlyActiveMinutes": "저강도 활동(분)",
    "SedentaryMinutes": "좌식 시간(분)",
}
FEAT_LIST = list(FEATURES.keys())
DEV_LIST = [f + "_dev" for f in FEAT_LIST]
COLOR_MAP = {"수면 부족": "#EF553B", "충분한 수면": "#636EFA"}

# 코드는 code 폴더(또는 최상위), 데이터는 data 폴더
HERE = Path(__file__).resolve().parent
DATA_DIRS = [HERE.parent / "data", HERE / "data"]


def find_file(keyword):
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.csv"):
            if keyword.lower() in p.name.lower():
                return p
    return None


# ------------------------------------------------------------------
# 1. 데이터 불러오기 (main.py와 동일: 오늘 활동 -> 다음 날 수면)
# ------------------------------------------------------------------
@st.cache_data
def load_data(act_path, sleep_path):
    act = pd.read_csv(act_path)
    sleep = pd.read_csv(sleep_path)
    act["ActivityDate"] = pd.to_datetime(act["ActivityDate"], format="%m/%d/%Y")
    sleep["SleepDay"] = pd.to_datetime(
        sleep["SleepDay"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()
    act["NextDay"] = act["ActivityDate"] + pd.Timedelta(days=1)
    df = act.merge(sleep, left_on=["Id", "NextDay"],
                   right_on=["Id", "SleepDay"], how="inner")
    df["Id"] = df["Id"].astype(str)
    return df


act_path = find_file("dailyactivity")
sleep_path = find_file("sleepday")
if act_path is None or sleep_path is None:
    st.error("data 폴더에서 dailyActivity / sleepDay CSV를 찾지 못했습니다. "
             f"확인한 위치: {[str(d) for d in DATA_DIRS]}")
    st.stop()
raw = load_data(str(act_path), str(sleep_path))

# ------------------------------------------------------------------
# 2. 사이드바 설정 + 전처리 (main.py와 같은 방식)
# ------------------------------------------------------------------
st.sidebar.header("⚙️ 설정")
threshold = st.sidebar.slider(
    "수면 부족 기준 (분 미만)", 300, 480, 360, 10,
    help="360분 = 6시간, 420분 = 7시간")
st.sidebar.caption(f"기준: {threshold // 60}시간 {threshold % 60}분 미만")
remove_noise = st.sidebar.checkbox("기기를 거의 안 찬 날 제외 (걸음 수 < 500)", value=True)
drop_short = st.sidebar.checkbox(
    "180분 미만 수면 기록 제외 (낮잠·부분 기록 의심)", value=False)
st.sidebar.caption(f"난수 고정: SEED={SEED}, 폴드 수={N_SPLITS}")

df = raw.copy()
n_raw = len(df)
if remove_noise:
    df = df[df["TotalSteps"] >= 500]
if drop_short:
    df = df[df["TotalMinutesAsleep"] >= 180]
df = df.reset_index(drop=True)

df["수면부족"] = (df["TotalMinutesAsleep"] < threshold).astype(int)
df["그룹"] = df["수면부족"].map({1: "수면 부족", 0: "충분한 수면"})

# 개인 평균 대비 편차 (필터링 후 계산)
for f in FEAT_LIST:
    df[f + "_dev"] = df[f] - df.groupby("Id")[f].transform("mean")
df["수면_dev"] = df["TotalMinutesAsleep"] - df.groupby("Id")["TotalMinutesAsleep"].transform("mean")

# 결측 처리: 모델에 쓰는 변수가 비어 있는 행은 제거
n_before = len(df)
df = df.dropna(subset=FEAT_LIST + DEV_LIST).reset_index(drop=True)
n_na = n_before - len(df)

if len(df) < 30 or df["Id"].nunique() < N_SPLITS or df["수면부족"].nunique() < 2:
    st.error("남은 데이터가 너무 적거나 한쪽 그룹만 있습니다. 사이드바 설정을 조정하세요.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.metric("분석 행 수", len(df))
st.sidebar.metric("참가자 수", df["Id"].nunique())
st.sidebar.metric("수면 부족 비율", f"{df['수면부족'].mean() * 100:.1f}%")
st.sidebar.caption(f"병합 후 {n_raw}행 → 필터·결측 제거 후 {len(df)}행 (결측 제거 {n_na}행)")

st.title("📊 박스플롯 vs 개인 내 변화 vs 베이스라인")
st.write("**질문: 하루의 활동 패턴만으로 다음 날 수면 부족을 예측할 수 있을까?**")

# ------------------------------------------------------------------
# 3. 그래프 두 개 (모델에 쓰는 두 가지 관점)
# ------------------------------------------------------------------
st.header("1. 두 가지 관점 살펴보기")
feat = st.selectbox("살펴볼 활동 변수", FEAT_LIST, format_func=lambda x: FEATURES[x])
left, right = st.columns(2)

with left:
    st.subheader("A. 수면 부족 그룹별 활동 (박스플롯)")
    fig_box = px.box(df, x="그룹", y=feat, color="그룹", points="all",
                     color_discrete_map=COLOR_MAP, labels={feat: FEATURES[feat]})
    st.plotly_chart(fig_box, use_container_width=True)

with right:
    st.subheader("B. 개인 내 변화 (평소 대비 편차)")
    x, y = df[feat + "_dev"].values, df["수면_dev"].values
    fig_sc = px.scatter(df, x=feat + "_dev", y="수면_dev", color="Id",
                        labels={feat + "_dev": f"평소 대비 {FEATURES[feat]} 차이",
                                "수면_dev": "평소 대비 다음 날 수면 차이(분)"})
    if np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        fig_sc.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                    name="추세선", line=dict(color="black", dash="dash")))
    fig_sc.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_sc.add_vline(x=0, line_dash="dot", line_color="gray")
    fig_sc.update_layout(showlegend=False)
    st.plotly_chart(fig_sc, use_container_width=True)


# ------------------------------------------------------------------
# 4. 같은 분할 · 같은 난수로 모델 비교
# ------------------------------------------------------------------
def make_splits(d):
    """모든 모델이 똑같이 쓰는 폴드. 같은 사람은 한 폴드에만 들어간다."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(cv.split(d[FEAT_LIST], d["수면부족"], groups=d["Id"]))


def evaluate(d, splits, feats, kind):
    """폴드별 예측을 모아 한 번에 지표를 계산한다."""
    y = d["수면부족"].values
    proba = np.zeros(len(d))
    pred = np.zeros(len(d), dtype=int)
    fold_aucs = []
    for tr, te in splits:
        if kind == "baseline":
            model = DummyClassifier(strategy="most_frequent")
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=SEED))
        model.fit(d.iloc[tr][feats], y[tr])
        p = model.predict_proba(d.iloc[te][feats])
        if p.shape[1] == 1:  # 한 클래스만 학습한 경우
            p1 = np.full(len(te), float(model.classes_[0] == 1))
        else:
            p1 = p[:, 1]
        proba[te] = p1
        pred[te] = model.predict(d.iloc[te][feats])
        if len(np.unique(y[te])) == 2:
            fold_aucs.append(roc_auc_score(y[te], p1))
    return {
        "정확도": accuracy_score(y, pred),
        "균형 정확도": balanced_accuracy_score(y, pred),
        "정밀도": precision_score(y, pred, zero_division=0),
        "재현율": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "AUC": roc_auc_score(y, proba),
        "폴드별 AUC 평균": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        "폴드별 AUC 표준편차": float(np.std(fold_aucs)) if fold_aucs else np.nan,
    }


st.header("2. 같은 분할 · 같은 변수 · 고정 난수로 비교 (지표 한 표)")
splits = make_splits(df)

results = {
    "베이스라인 (다수 범주 찍기)": evaluate(df, splits, FEAT_LIST, "baseline"),
    "A. 활동 원래 값 (박스플롯 관점)": evaluate(df, splits, FEAT_LIST, "model"),
    "B. 평소 대비 편차 (개인 내 변화 관점)": evaluate(df, splits, DEV_LIST, "model"),
}
table = pd.DataFrame(results).T.round(3)
table.index.name = "모델"
st.dataframe(table, use_container_width=True)

maj = "충분한 수면" if df["수면부족"].mean() < 0.5 else "수면 부족"
st.caption(
    f"모든 모델: 같은 {len(df)}행, 같은 5개 변수, 같은 {N_SPLITS}폴드(StratifiedGroupKFold, SEED={SEED}). "
    f"베이스라인은 항상 '{maj}'만 예측하므로 정확도는 높아 보여도 AUC·균형 정확도는 0.5입니다. "
    "재현율이 0이면 수면 부족을 하나도 못 찾았다는 뜻입니다.")

st.info("💭 생각해 볼 질문\n"
        "- 모델 A, B는 베이스라인보다 AUC·균형 정확도가 의미 있게 높은가요?\n"
        "- A와 B의 차이는 폴드별 AUC 표준편차보다 큰가요, 작은가요?\n"
        "- 정확도만 보면 베이스라인이 좋아 보일 수 있는데, 다른 지표는 무엇을 알려 주나요?\n"
        "- 기준(360/420분)이나 제외 옵션을 바꿔도 같은 결론인가요?")

with st.expander("⚠️ 이 비교의 한계"):
    st.markdown(
        """
- 참가자 수가 적어 폴드마다 성능이 크게 흔들립니다(폴드별 AUC 표준편차 확인).
- 모델 B의 '개인 평균'은 그 사람의 모든 날 활동으로 계산해 완전한 미래 정보 없는 예측은 아닙니다.
- 좌식 시간에는 수면 시간 정보가 섞였을 수 있습니다.
- 이상치는 별도로 보정하지 않고, 사이드바의 제외 옵션으로만 다룹니다.
- 이 코드는 상관·예측 성능만 보여 주며, 인과관계를 뜻하지 않습니다.
        """
    )
