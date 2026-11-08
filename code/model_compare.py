import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, balanced_accuracy_score)

# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(page_title="모델 비교", layout="wide")
st.title("🧪 베이스라인 vs 후보 모델 비교")
st.caption("질문: 하루의 활동 패턴만으로 다음 날 수면 부족 여부를 예측할 수 있는가?")

SEED = 42

# 경로: 이 파일(code/) 기준으로 상위 폴더의 data/ 를 가리킴
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
ACTIVITY_PATH = os.path.join(DATA_DIR, "dailyActivity_merged.csv")
SLEEP_PATH = os.path.join(DATA_DIR, "sleepDay_merged.csv")

FEATURES = ["TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes",
            "LightlyActiveMinutes", "SedentaryMinutes"]
METRICS = ["정확도", "정밀도", "재현율", "F1", "AUC", "균형정확도"]

BASELINES = ["Dummy(다수 클래스)", "로지스틱 회귀"]
CANDIDATES = ["랜덤 포레스트", "그래디언트 부스팅", "KNN"]


def get_models():
    return {
        "Dummy(다수 클래스)": DummyClassifier(strategy="most_frequent"),
        "로지스틱 회귀": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        "랜덤 포레스트": RandomForestClassifier(
            n_estimators=200, max_depth=4, class_weight="balanced", random_state=SEED),
        "그래디언트 부스팅": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, random_state=SEED),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    }


# ----------------------------
# 데이터 불러오기
# ----------------------------
@st.cache_data
def load_activity(path):
    df = pd.read_csv(path)
    df["Id"] = df["Id"].astype(str)
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], format="%m/%d/%Y")
    df["NextDate"] = df["ActivityDate"] + pd.Timedelta(days=1)
    return df


@st.cache_data
def load_sleep(path_or_file):
    sl = pd.read_csv(path_or_file)
    sl["Id"] = sl["Id"].astype(str)
    sl["SleepDate"] = pd.to_datetime(sl["SleepDay"], errors="coerce").dt.normalize()
    return sl


def evaluate(model, X_tr, y_tr, X_te, y_te):
    m = clone(model).fit(X_tr, y_tr)
    proba = m.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "정확도": accuracy_score(y_te, pred),
        "정밀도": precision_score(y_te, pred, zero_division=0),
        "재현율": recall_score(y_te, pred, zero_division=0),
        "F1": f1_score(y_te, pred, zero_division=0),
        "AUC": roc_auc_score(y_te, proba),
        "균형정확도": balanced_accuracy_score(y_te, pred),
    }


def size_label(diff, std):
    """차이의 크기를 말로 표현 (대략적인 기준, 절대적 규칙 아님)"""
    if diff <= 0:
        return "베이스라인보다 낫지 않음"
    if diff <= std:
        return "차이가 작음 (분할마다 흔들리는 폭 안쪽이라 우연일 수 있음)"
    if diff < 0.05:
        return "작은 편"
    if diff < 0.10:
        return "중간 정도"
    return "큰 편"


# ----------------------------
# 데이터 준비
# ----------------------------
if not os.path.exists(ACTIVITY_PATH):
    st.error(f"활동 데이터를 찾을 수 없습니다: {os.path.normpath(ACTIVITY_PATH)}")
    st.stop()
act = load_activity(ACTIVITY_PATH)

if os.path.exists(SLEEP_PATH):
    sleep = load_sleep(SLEEP_PATH)
    st.success("data/sleepDay_merged.csv 를 불러왔어요.")
else:
    st.warning("data/sleepDay_merged.csv 가 없어요. 파일을 직접 업로드해 주세요.")
    sleep_file = st.file_uploader("sleepDay_merged.csv 업로드", type="csv")
    if sleep_file is None:
        st.stop()
    sleep = load_sleep(sleep_file)

data = act.merge(
    sleep[["Id", "SleepDate", "TotalMinutesAsleep"]],
    left_on=["Id", "NextDate"], right_on=["Id", "SleepDate"], how="inner",
)

st.sidebar.header("⚙️ 설정")
threshold = st.sidebar.slider("수면 부족 기준(분)", 300, 540, 420, 30)
min_steps = st.sidebar.slider("최소 걸음 수 (기기 미착용 제거)", 0, 3000, 500, 100)
test_size = st.sidebar.slider("시험 데이터 비율(사용자 기준)", 0.2, 0.5, 0.3, 0.05)
n_repeats = st.sidebar.slider("반복 분할 횟수", 5, 50, 20, 5)
st.sidebar.write(f"난수 고정값: **{SEED}**")

data = data[data["TotalSteps"] >= min_steps].copy()
data["label"] = (data["TotalMinutesAsleep"] < threshold).astype(int)  # 1 = 수면 부족

st.write(f"사용 데이터: **{len(data)}건**, 사용자 **{data['Id'].nunique()}명**, "
         f"수면 부족 비율 **{data['label'].mean():.1%}**")

if data["label"].nunique() < 2 or len(data) < 50 or data["Id"].nunique() < 6:
    st.warning("라벨이 한쪽뿐이거나 데이터가 너무 적습니다. 기준을 조정해 보세요.")
    st.stop()

X = data[FEATURES].values
y = data["label"].values
groups = data["Id"].values
models = get_models()

# ----------------------------
# ① 고정 분할 1회 (모든 모델 동일 분할)
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
tr_idx, te_idx = next(gss.split(X, y, groups))

if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
    st.warning("고정 분할의 학습 또는 시험 데이터에 한 클래스만 있어요. 기준이나 시험 비율을 바꿔보세요.")
    st.stop()

single_rows = []
for name, model in models.items():
    r = evaluate(model, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
    r["모델"] = name
    r["구분"] = "베이스라인" if name in BASELINES else "후보"
    single_rows.append(r)
single = pd.DataFrame(single_rows)[["구분", "모델"] + METRICS].round(3)

st.subheader("📊 ① 고정 분할 결과 (베이스라인 + 후보 한 표)")
st.write(f"학습 {len(tr_idx)}건 / 시험 {len(te_idx)}건 "
         f"(시험 사용자 {len(np.unique(groups[te_idx]))}명, 학습 사용자와 겹치지 않음)")
st.dataframe(single, use_container_width=True, hide_index=True)

# ----------------------------
# ② 반복 분할 (같은 분할을 모든 모델에 적용)
# ----------------------------
rep_rows = []
gss_rep = GroupShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=SEED)
valid_splits = 0
for i, (tr, te) in enumerate(gss_rep.split(X, y, groups)):
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        continue  # 한 클래스뿐인 분할은 AUC 계산 불가 → 모든 모델에서 동일하게 제외
    valid_splits += 1
    for name, model in models.items():
        r = evaluate(model, X[tr], y[tr], X[te], y[te])
        r["모델"] = name
        r["분할"] = i
        rep_rows.append(r)

if valid_splits == 0:
    st.warning("유효한 반복 분할이 없습니다. 설정을 바꿔보세요.")
    st.stop()

rep = pd.DataFrame(rep_rows)
mean_df = rep.groupby("모델")[METRICS].mean()
std_df = rep.groupby("모델")[METRICS].std()

summary = mean_df.copy().round(3).astype(str) + " ± " + std_df.round(3).astype(str)
summary = summary.loc[BASELINES + CANDIDATES].reset_index()
summary.insert(0, "구분", ["베이스라인"] * len(BASELINES) + ["후보"] * len(CANDIDATES))

st.subheader(f"📊 ② 반복 분할 결과 (유효 분할 {valid_splits}회, 평균 ± 표준편차)")
st.dataframe(summary, use_container_width=True, hide_index=True)
st.caption("같은 분할에서 모든 모델이 시험을 봤어요. ±는 분할이 바뀔 때 점수가 흔들리는 폭이에요.")

# ----------------------------
# 그래프
# ----------------------------
st.subheader("📈 결과 그래프")
long_mean = mean_df.loc[BASELINES + CANDIDATES].reset_index().melt(
    id_vars="모델", var_name="지표", value_name="점수")
st.plotly_chart(
    px.bar(long_mean, x="지표", y="점수", color="모델", barmode="group",
           category_orders={"모델": BASELINES + CANDIDATES},
           title="지표별 평균 점수 (반복 분할)", range_y=[0, 1]),
    use_container_width=True,
)

st.plotly_chart(
    px.box(rep, x="모델", y="AUC", color="모델", points="all",
           category_orders={"모델": BASELINES + CANDIDATES},
           title="분할마다 AUC가 얼마나 흔들리는가 (0.5 = 동전 던지기)")
    .add_hline(y=0.5, line_dash="dash"),
    use_container_width=True,
)

# ----------------------------
# 해석 ① 지표가 우리 데이터에서 의미하는 것
# ----------------------------
st.header("📖 결과 해석")
prev = data["label"].mean()

st.subheader("1. 각 지표는 우리 데이터에서 무슨 뜻인가?")
st.markdown(f"""
(여기서 **양성 = 수면 부족**, 수면 부족 비율은 현재 **{prev:.1%}**)

| 지표 | 우리 데이터에서의 의미 | 왜 중요하게 보는가 |
|---|---|---|
| **정확도** | 전체 (활동일 → 다음 날) 쌍 중 부족/충분을 맞힌 비율 | 직관적이지만 한쪽 클래스가 많으면 속기 쉬워요. 항상 다수 쪽으로 찍어도 약 {max(prev, 1-prev):.0%}가 나와요. |
| **정밀도** | "내일 수면 부족"이라고 경고한 날 중 실제로 부족했던 비율 | 오경보를 얼마나 줄이는지 보여줘요. 낮으면 경고를 믿기 어려워요. |
| **재현율** | 실제 수면 부족이었던 날 중 모델이 잡아낸 비율 | 수면 부족을 놓치지 않는 능력이에요. "항상 부족"이라고 찍으면 100%가 되니 정밀도와 같이 봐야 해요. |
| **F1** | 정밀도와 재현율의 조화평균 | 두 지표 중 하나만 높은 속임수를 막아줘요. 클래스가 쏠린 데이터에서 정확도보다 믿을 만해요. |
| **AUC** | 수면 부족인 날과 충분한 날을 무작위로 하나씩 뽑았을 때, 부족인 날에 더 높은 위험 점수를 줄 확률. 0.5는 동전 던지기, 1은 완벽 | 기준 확률(0.5)에 상관없이 "활동 패턴에 구분 정보가 있는가"를 보여줘요. 우리 질문에 가장 직접적이에요. |
| **균형정확도** | 부족일 때 맞힌 비율(재현율)과 충분일 때 맞힌 비율의 평균 | 클래스 비율이 달라도 공정해요. Dummy는 항상 0.5가 돼요. |
""")

# ----------------------------
# 해석 ② 베이스라인과의 차이
# ----------------------------
st.subheader("2. 베이스라인보다 나은 모델이 있는가? 차이는 큰가?")

dummy_name, logit_name = BASELINES
rows = []
for m in CANDIDATES:
    for metric in ["F1", "AUC", "균형정확도"]:
        diff_d = (rep[rep["모델"] == m].set_index("분할")[metric]
                  - rep[rep["모델"] == dummy_name].set_index("분할")[metric])
        diff_l = (rep[rep["모델"] == m].set_index("분할")[metric]
                  - rep[rep["모델"] == logit_name].set_index("분할")[metric])
        rows.append({
            "후보 모델": m, "지표": metric,
            "Dummy 대비 평균 차이": round(diff_d.mean(), 3),
            "로지스틱 대비 평균 차이": round(diff_l.mean(), 3),
            "차이의 표준편차": round(diff_l.std(), 3),
            "판단(로지스틱 대비)": size_label(diff_l.mean(), diff_l.std()),
        })
cmp_df = pd.DataFrame(rows)
st.dataframe(cmp_df, use_container_width=True, hide_index=True)

best_auc = mean_df.loc[CANDIDATES, "AUC"].idxmax()
best_auc_val = mean_df.loc[best_auc, "AUC"]
auc_std = std_df.loc[best_auc, "AUC"]
auc_gain = best_auc_val - 0.5

st.markdown(f"""
**자동 요약**
- AUC가 가장 높은 후보는 **{best_auc}** (평균 AUC **{best_auc_val:.3f}**, 표준편차 {auc_std:.3f}).
- 동전 던지기(0.5)보다 **{auc_gain:+.3f}** 높아요 → **{size_label(auc_gain, auc_std)}**.
- 이 판단은 "차이가 분할마다 흔들리는 폭(표준편차)보다 큰가?"와 대략적인 크기 기준(0.05, 0.10)으로 한 거예요. 통계적 검정이 아닌 **참고용 기준**이에요.

**읽는 법**
- 평균 차이가 표준편차보다 작으면, 분할을 바꿨을 때 순위가 뒤바뀔 수 있어서 "더 낫다"고 말하기 어려워요.
- 사용자 수가 적으면(약 24명) 어떤 사용자가 시험에 들어가느냐에 따라 점수가 크게 달라져요. 표준편차가 큰 것 자체가 중요한 결과예요.
""")

# ----------------------------
# 해석 ③ 지표에 따라 결론이 달라지는가
# ----------------------------
st.subheader("3. 다른 지표로 보면 결론이 달라지는가?")

rank_rows = []
for metric in METRICS:
    ordered = mean_df[metric].sort_values(ascending=False)
    rank_rows.append({"지표": metric, "1위 모델": ordered.index[0],
                      "1위 점수": round(ordered.iloc[0], 3),
                      "Dummy 점수": round(mean_df.loc[dummy_name, metric], 3)})
rank_df = pd.DataFrame(rank_rows)
st.dataframe(rank_df, use_container_width=True, hide_index=True)

n_unique_winners = rank_df["1위 모델"].nunique()
acc_winner = rank_df.loc[rank_df["지표"] == "정확도", "1위 모델"].iloc[0]
auc_winner = rank_df.loc[rank_df["지표"] == "AUC", "1위 모델"].iloc[0]
dummy_acc = mean_df.loc[dummy_name, "정확도"]
dummy_f1 = mean_df.loc[dummy_name, "F1"]

if n_unique_winners == 1:
    verdict = f"모든 지표에서 1위가 **{rank_df['1위 모델'].iloc[0]}**로 같아요. 지표에 따라 결론이 달라지지 않았어요."
else:
    verdict = (f"지표마다 1위가 달라요 (서로 다른 1위 모델 {n_unique_winners}개). "
               f"정확도 1위는 **{acc_winner}**, AUC 1위는 **{auc_winner}**예요. "
               "따라서 어떤 지표를 기준으로 삼느냐에 따라 '최고의 모델'이 달라질 수 있어요.")

st.markdown(f"""
{verdict}

**주의해서 볼 점**
- Dummy의 정확도는 **{dummy_acc:.3f}**인데 F1은 **{dummy_f1:.3f}**예요. 정확도만 보면 쓸 만해 보이는데, 실제로는 아무 정보도 쓰지 않은 모델이에요.
- 재현율만 보면 "항상 수면 부족"이라고 찍는 모델이 유리하고, 정밀도만 보면 아주 조심스럽게 찍는 모델이 유리해요. 그래서 **F1, AUC, 균형정확도를 같이** 봐야 해요.
- 모든 지표에서 후보가 베이스라인과 비슷하다면, 연구 결론은 **"이 활동 변수들만으로는 다음 날 수면 부족을 신뢰할 만큼 예측하기 어렵다"**가 돼요. 이것도 근거 있는 결과예요.
""")

st.info(
    "한계: 사용자 수와 데이터 수가 적고, 수면 부족 기준(분)이나 기기 미착용 제거 기준을 바꾸면 결과가 달라질 수 있어요. "
    "사이드바의 기준을 바꿔서 결론이 일관적인지 꼭 확인해 보세요. "
    "그래디언트 부스팅과 KNN은 클래스 비율 보정을 쓰지 않아 재현율이 낮게 나올 수 있어요."
)
