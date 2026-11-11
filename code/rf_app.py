import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, balanced_accuracy_score,
                             confusion_matrix, roc_curve)

st.set_page_config(page_title="랜덤 포레스트 수면 부족 예측", layout="wide")

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
ACTIVITY_PATH = os.path.join(DATA_DIR, "dailyActivity_merged.csv")
SLEEP_PATH = os.path.join(DATA_DIR, "sleepDay_merged.csv")

FEATURES = ["TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes",
            "LightlyActiveMinutes", "SedentaryMinutes"]
FEATURE_KR = {
    "TotalSteps": "걸음 수",
    "VeryActiveMinutes": "매우 활발한 활동(분)",
    "FairlyActiveMinutes": "적당히 활발한 활동(분)",
    "LightlyActiveMinutes": "가벼운 활동(분)",
    "SedentaryMinutes": "좌식(분)",
}
GRID = {"n_estimators": [100, 300],
        "max_depth": [2, 3, 4, 6],
        "min_samples_leaf": [5, 10, 20]}


# ---------------- 데이터 / 모델 함수 ----------------
@st.cache_data
def load_dataset(threshold, min_steps):
    act = pd.read_csv(ACTIVITY_PATH)
    act["Id"] = act["Id"].astype(str)
    act["ActivityDate"] = pd.to_datetime(act["ActivityDate"], format="%m/%d/%Y")
    act["NextDate"] = act["ActivityDate"] + pd.Timedelta(days=1)
    sl = pd.read_csv(SLEEP_PATH)
    sl["Id"] = sl["Id"].astype(str)
    sl["SleepDate"] = pd.to_datetime(sl["SleepDay"], errors="coerce").dt.normalize()
    data = act.merge(sl[["Id", "SleepDate", "TotalMinutesAsleep"]],
                     left_on=["Id", "NextDate"], right_on=["Id", "SleepDate"],
                     how="inner")
    data = data[data["TotalSteps"] >= min_steps].copy()
    data["label"] = (data["TotalMinutesAsleep"] < threshold).astype(int)
    return data.reset_index(drop=True)


def calc_metrics(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "정확도": accuracy_score(y_true, pred),
        "정밀도": precision_score(y_true, pred, zero_division=0),
        "재현율": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "AUC": roc_auc_score(y_true, proba),
        "균형정확도": balanced_accuracy_score(y_true, pred),
    }


@st.cache_data(show_spinner="학습/시험 데이터를 사용자 단위로 나누는 중...")
def get_split(threshold, min_steps, test_size):
    data = load_dataset(threshold, min_steps)
    if data["label"].nunique() < 2 or len(data) < 50 or data["Id"].nunique() < 6:
        return None
    y = data["label"].values
    groups = data["Id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr, te = next(gss.split(data[FEATURES].values, y, groups))
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    return data, tr, te


@st.cache_data(show_spinner="최적 하이퍼파라미터를 찾는 중 (교차검증)...")
def find_best_params(threshold, min_steps, test_size):
    split = get_split(threshold, min_steps, test_size)
    if split is None:
        return None
    data, tr, _ = split
    X = data[FEATURES].values
    y = data["label"].values
    groups = data["Id"].values
    base_rf = RandomForestClassifier(class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
    n_groups = len(np.unique(groups[tr]))
    gs = GridSearchCV(base_rf, GRID, scoring="roc_auc",
                      cv=GroupKFold(n_splits=min(4, n_groups)), n_jobs=-1)
    gs.fit(X[tr], y[tr], groups=groups[tr])
    cv_table = (pd.DataFrame(gs.cv_results_)
                [["param_n_estimators", "param_max_depth", "param_min_samples_leaf",
                  "mean_test_score", "std_test_score"]]
                .rename(columns={"param_n_estimators": "트리 수",
                                 "param_max_depth": "최대 깊이",
                                 "param_min_samples_leaf": "리프 최소 표본",
                                 "mean_test_score": "교차검증 AUC 평균",
                                 "std_test_score": "표준편차"})
                .sort_values("교차검증 AUC 평균", ascending=False)
                .reset_index(drop=True))
    best = {k: int(v) for k, v in gs.best_params_.items()}
    return {"best_params": best, "cv_table": cv_table}


@st.cache_data(show_spinner="랜덤 포레스트를 학습하는 중...")
def train(threshold, min_steps, test_size, n_estimators, max_depth, min_samples_leaf):
    split = get_split(threshold, min_steps, test_size)
    if split is None:
        return None
    data, tr, te = split
    X = data[FEATURES].values
    y = data["label"].values
    groups = data["Id"].values
    params = {"n_estimators": n_estimators, "max_depth": max_depth,
              "min_samples_leaf": min_samples_leaf}
    model = RandomForestClassifier(class_weight="balanced", random_state=SEED,
                                   n_jobs=-1, **params).fit(X[tr], y[tr])

    n_groups = len(np.unique(groups[tr]))
    cv_scores = cross_val_score(
        RandomForestClassifier(class_weight="balanced", random_state=SEED,
                               n_jobs=-1, **params),
        X[tr], y[tr], groups=groups[tr], scoring="roc_auc",
        cv=GroupKFold(n_splits=min(4, n_groups)), error_score=np.nan)
    all_nan = bool(np.all(np.isnan(cv_scores)))
    cv_mean = np.nan if all_nan else float(np.nanmean(cv_scores))
    cv_std = np.nan if all_nan else float(np.nanstd(cv_scores))

    dummy = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
    rf_proba = model.predict_proba(X[te])[:, 1]
    dm_proba = dummy.predict_proba(X[te])[:, 1]
    train_proba = model.predict_proba(X[tr])[:, 1]
    scores = pd.DataFrame({
        "Dummy(다수 클래스)": calc_metrics(y[te], dm_proba),
        "랜덤 포레스트": calc_metrics(y[te], rf_proba),
    }).T

    perm = permutation_importance(model, X[te], y[te], scoring="roc_auc",
                                  n_repeats=30, random_state=SEED, n_jobs=-1)
    importance = pd.DataFrame({
        "변수": [FEATURE_KR[f] for f in FEATURES],
        "트리 중요도": model.feature_importances_,
        "순열 중요도": perm.importances_mean,
        "순열 표준편차": perm.importances_std,
    })
    fpr, tpr, _ = roc_curve(y[te], rf_proba)
    return {
        "data": data, "model": model, "params": params,
        "cv_mean": cv_mean, "cv_std": cv_std,
        "train_auc": float(roc_auc_score(y[tr], train_proba)),
        "scores": scores, "importance": importance,
        "cm": confusion_matrix(y[te], (rf_proba >= 0.5).astype(int)),
        "fpr": fpr, "tpr": tpr,
        "n_train": len(tr), "n_test": len(te),
        "train_users": len(np.unique(groups[tr])),
        "test_users": len(np.unique(groups[te])),
        "test_prev": float(y[te].mean()),
        "train_prev": float(y[tr].mean()),
        "proba_std": float(np.std(rf_proba)),
    }


# ---------------- 사이드바 ----------------
st.sidebar.header("⚙️ 설정")
st.sidebar.subheader("데이터 설정")
threshold = st.sidebar.slider("수면 부족 기준(분)", 300, 540, 420, 30,
                              help="이 값보다 적게 자면 '수면 부족'. 420분 = 7시간")
min_steps = st.sidebar.slider("최소 걸음 수", 0, 3000, 500, 100,
                              help="기기를 거의 안 찬 날을 제외")
test_size = st.sidebar.slider("시험 데이터 비율(사용자 기준)", 0.2, 0.5, 0.3, 0.05,
                              help="전체 사용자 중 시험용으로 빼두는 비율")

if not (os.path.exists(ACTIVITY_PATH) and os.path.exists(SLEEP_PATH)):
    st.error("data 폴더에서 dailyActivity_merged.csv 또는 sleepDay_merged.csv를 찾을 수 없어요. "
             f"확인한 경로: {os.path.normpath(DATA_DIR)}")
    st.stop()

tuned = find_best_params(threshold, min_steps, test_size)

st.sidebar.subheader("모델 설정 (하이퍼파라미터)")
if tuned is None:
    mode = "자동(최적값)"
    best = {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 10}
    chosen = best
    st.sidebar.info("데이터가 부족해 튜닝을 못 했어요. 기본 설정을 써요.")
else:
    best = tuned["best_params"]
    mode = st.sidebar.radio(
        "설정 방식", ["자동(최적값)", "직접 설정"], index=0,
        help="자동: 교차검증 최적 조합 / 직접 설정: 슬라이더 값(시작값은 최적값)")
    if mode == "자동(최적값)":
        chosen = best
        st.sidebar.success(f"최적값 → 트리 {best['n_estimators']}개 / "
                           f"깊이 {best['max_depth']} / 리프 {best['min_samples_leaf']}")
    else:
        tag = f"{best['n_estimators']}_{best['max_depth']}_{best['min_samples_leaf']}"
        n_est = st.sidebar.slider("트리 수", 50, 500, best["n_estimators"], 50,
                                  key=f"n_{tag}", help="투표하는 트리 개수")
        depth = st.sidebar.slider("최대 깊이", 1, 12, best["max_depth"], 1,
                                  key=f"d_{tag}", help="클수록 복잡해져요")
        leaf = st.sidebar.slider("리프 최소 표본", 1, 50, best["min_samples_leaf"], 1,
                                 key=f"l_{tag}", help="클수록 단순해져요")
        chosen = {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": leaf}
st.sidebar.caption(f"난수 고정값: {SEED}")

# ---------------- 제목 ----------------
st.title("🌲 랜덤 포레스트로 다음 날 수면 부족 예측하기")
st.markdown("""
**연구 질문**: 하루의 활동 패턴(걸음 수, 활동 시간)만으로 **다음 날 수면 부족 여부**를 예측할 수 있는가?

> 랜덤 포레스트는 "걸음 수가 많은가?" 같은 질문으로 나뉘는 **결정 트리 여러 개가 투표**하는 모델이에요.
> 직선 관계뿐 아니라 U자형·조합 관계도 찾을 수 있어요.
""")

res = train(threshold, min_steps, test_size,
            chosen["n_estimators"], chosen["max_depth"], chosen["min_samples_leaf"])
if res is None:
    st.warning("데이터가 너무 적거나 한쪽 라벨만 있어요. 사이드바의 기준을 조정해 보세요.")
    st.stop()

data = res["data"]
prev = data["label"].mean()
sc = res["scores"]
rf = sc.loc["랜덤 포레스트"]
dm = sc.loc["Dummy(다수 클래스)"]
auc = rf["AUC"]
gap = res["train_auc"] - auc
majority = max(res["test_prev"], 1 - res["test_prev"])
bp = res["params"]

# ---------------- 0. 데이터 ----------------
st.header("0️⃣ 데이터 한눈에 보기")
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 데이터(건)", f"{len(data)}")
c2.metric("사용자 수(명)", f"{data['Id'].nunique()}")
c3.metric("수면 부족 비율", f"{prev:.1%}")
c4.metric("학습 / 시험(건)", f"{res['n_train']} / {res['n_test']}")

avg_per_user = len(data) / data["Id"].nunique()
prev_msg = ("한쪽으로 치우쳐 있어요. 정확도보다 AUC·균형정확도를 보세요."
            if min(prev, 1 - prev) < 0.2 else "두 클래스가 비교적 균형 잡혀 있어요.")
with st.expander("📘 이 숫자들은 무슨 뜻이에요?", expanded=True):
    st.markdown(f"""
- **전체 {len(data)}건**: '오늘 활동 → 내일 수면' 쌍이에요. 두 기록이 모두 있는 날만 포함돼요.
- **사용자 {data['Id'].nunique()}명, 1인당 평균 {avg_per_user:.1f}건**: 같은 사람의 데이터는 비슷해서 독립 정보량은 건수보다 **사람 수**에 가까워요.
- **수면 부족 비율 {prev:.1%}**: 다음 날 {threshold}분({threshold/60:.1f}시간) 미만 수면 비율. {prev_msg}
- **학습/시험**: 사용자 단위 분리(학습 {res['train_users']}명, 시험 {res['test_users']}명, 겹치지 않음). 겹치면 습관을 외워 점수가 부풀려져요.
- **부족 비율** 학습 {res['train_prev']:.1%} / 시험 {res['test_prev']:.1%}: 많이 다르면 사용자 구성 때문에 점수가 흔들려요.
""")

st.subheader("수면 시간 분포")
fig_hist = px.histogram(data, x="TotalMinutesAsleep", nbins=30,
                        labels={"TotalMinutesAsleep": "다음 날 수면 시간(분)"},
                        title="수면 시간 분포와 '수면 부족' 기준선")
fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                   annotation_text=f"기준 {threshold}분")
st.plotly_chart(fig_hist, use_container_width=True)
median_sleep = data["TotalMinutesAsleep"].median()
near = int(((data["TotalMinutesAsleep"] - threshold).abs() <= 30).sum())
st.markdown(f"""
- 수면 **중앙값 {median_sleep:.0f}분({median_sleep/60:.1f}시간)**. 기준선이 중앙에 가까울수록 두 클래스가 비슷하게 나뉘어요.
- 기준을 옮기면 라벨이 바뀌어 **문제 난이도가 달라져요.**
- 기준 ±30분 안 데이터 **{near}건({near/len(data):.1%})**: 거의 같은 수면이 다른 라벨을 받아 예측이 어려워져요.
""")

st.subheader("수면 부족인 날 vs 충분한 날, 전날 활동 평균")
comp = (data.groupby("label")[FEATURES].mean().T
        .rename(columns={0: "수면 충분", 1: "수면 부족"}))
comp["차이(부족-충분)"] = comp["수면 부족"] - comp["수면 충분"]
comp["차이 비율(%)"] = comp["차이(부족-충분)"] / comp["수면 충분"].replace(0, np.nan) * 100
comp.index = [FEATURE_KR[f] for f in comp.index]
st.dataframe(comp.round(1), use_container_width=True)
biggest = comp["차이 비율(%)"].abs().idxmax()
st.markdown(f"""
- 모델 이전의 **기초 비교**예요. 차이 비율이 가장 큰 변수: **{biggest}**({comp.loc[biggest, '차이 비율(%)']:+.1f}%).
- ⚠️ 사람마다 퍼짐이 크면 평균 차이는 우연일 수 있어요. ±5% 이내면 사실상 같아요.
- ⚠️ **상관**일 뿐 **원인**이 아니에요. 바쁜 생활이 활동과 수면 둘 다에 영향을 줬을 수 있어요.
""")

# ---------------- 1. 성능 ----------------
st.header("1️⃣ 시험 성적: 랜덤 포레스트 vs 찍기(Dummy)")
st.caption("모델이 본 적 없는 사용자로 측정한 점수예요. 초록색이 지표별 더 높은 모델이에요.")
st.dataframe(sc.round(3).style.highlight_max(axis=0, color="#d4edda"),
             use_container_width=True)
fig_perf = go.Figure()
for name in sc.index:
    fig_perf.add_trace(go.Bar(name=name, x=sc.columns.tolist(), y=sc.loc[name].values))
fig_perf.update_layout(barmode="group", yaxis=dict(range=[0, 1]),
                       title="지표별 점수 비교 (높을수록 좋음)")
st.plotly_chart(fig_perf, use_container_width=True)

if auc < 0.55:
    verdict = "🔴 **동전 던지기 수준**이에요. 단서를 거의 못 찾았어요."
elif auc < 0.65:
    verdict = "🟡 **약한 신호**예요. 실제 예측엔 부족하고 우연일 수도 있어요."
elif auc < 0.8:
    verdict = "🟢 **어느 정도 구분**해요. 시험 사용자가 적어 반복 확인이 필요해요."
else:
    verdict = "🟢 **꽤 잘 구분**해요. 너무 좋다면 누수나 우연을 먼저 의심하세요."
st.info(f"랜덤 포레스트 AUC **{auc:.3f}** → {verdict}")

with st.expander("📘 각 지표는 무슨 뜻이고, 우리 결과는 어때요?", expanded=True):
    st.markdown(f"""
| 지표 | 의미 | 랜덤 포레스트 | Dummy |
|---|---|---|---|
| **AUC** ⭐ | 부족인 날과 충분한 날을 하나씩 뽑았을 때 부족인 날에 더 높은 점수를 줄 확률(0.5=동전) | **{rf['AUC']:.3f}** | {dm['AUC']:.3f} |
| **균형정확도** | 부족을 맞힌 비율과 충분을 맞힌 비율의 평균 | {rf['균형정확도']:.3f} | {dm['균형정확도']:.3f} |
| **F1** | 정밀도와 재현율의 조화평균 | {rf['F1']:.3f} | {dm['F1']:.3f} |
| **정밀도** | '부족'이라 경고한 날 중 실제 부족 비율 | {rf['정밀도']:.3f} | {dm['정밀도']:.3f} |
| **재현율** | 실제 부족한 날 중 잡아낸 비율 | {rf['재현율']:.3f} | {dm['재현율']:.3f} |
| **정확도** | 전체 중 맞힌 비율 | {rf['정확도']:.3f} | {dm['정확도']:.3f} |

- **AUC·균형정확도**가 가장 중요해요. Dummy는 항상 같은 답을 내서 둘 다 0.5예요.
- **정확도 주의**: 시험 부족 비율이 {res['test_prev']:.1%}라 다수만 찍어도 약 {majority:.0%}예요. 모델 정확도 {rf['정확도']:.1%}는 찍기 대비 **{(rf['정확도'] - majority)*100:+.1f}%p**.
- **정밀도**가 기본 비율({res['test_prev']:.1%})과 비슷하면 경고에 새 정보가 없어요. 현재 차이 **{(rf['정밀도'] - res['test_prev'])*100:+.1f}%p**.
- **재현율만 높으면** 거의 다 '부족'이라 찍었을 수 있어요. 정밀도와 함께 보세요.
- AUC는 0.5보다 **{auc - 0.5:+.3f}**, 균형정확도는 0.5보다 **{rf['균형정확도'] - 0.5:+.3f}**. 이 값이 핵심이에요.
- `class_weight="balanced"` 때문에 적은 클래스를 더 중시해서 재현율↑ 정밀도↓ 경향이 있어요.
- ⚠️ 시험 사용자 {res['test_users']}명뿐이라 다른 사용자가 들어가면 점수가 크게 변할 수 있어요.
""")

st.subheader("과적합 진단")
d1, d2, d3, d4 = st.columns(4)
d1.metric("학습 AUC", f"{res['train_auc']:.3f}")
d2.metric("교차검증 AUC", "—" if np.isnan(res["cv_mean"]) else f"{res['cv_mean']:.3f}")
d3.metric("시험 AUC", f"{auc:.3f}")
d4.metric("학습-시험 차이", f"{gap:+.3f}")
if gap > 0.25:
    gap_msg = "🔴 **심한 과적합**: 외우기만 했어요. 깊이를 줄이거나 리프 최소 표본을 늘려 보세요."
elif gap > 0.1:
    gap_msg = "🟡 **어느 정도 과적합**: 모델을 더 단순하게 해 보세요."
else:
    gap_msg = "🟢 차이가 작아요. 다만 **둘 다 낮다면** 과적합이 아니라 변수에 신호가 부족한 거예요."
cv_std_txt = "—" if np.isnan(res["cv_std"]) else f"{res['cv_std']:.3f}"
st.markdown(f"""
- **학습 AUC**는 이미 본 데이터 점수라 높은 게 정상이에요. 성능 증거로 쓰지 마세요.
- **교차검증 AUC**는 학습 데이터 안에서 사용자를 나눠 평가한 시험 점수의 예고편이에요(표준편차 {cv_std_txt}, 클수록 불안정).
- **시험 AUC**가 가장 믿을 만해요.
- **차이 {gap:+.3f}**: {gap_msg}
- 교차검증과 시험 AUC가 많이 다르면 시험 사용자 구성이 우연히 쉽거나 어려웠을 수 있어요.
""")

# ---------------- 2. ROC ----------------
st.header("2️⃣ ROC 곡선")
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=res["fpr"], y=res["tpr"], mode="lines",
                             name=f"랜덤 포레스트 (AUC {auc:.3f})", line=dict(width=3)))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="찍기 (0.5)",
                             line=dict(dash="dash", color="gray")))
fig_roc.update_layout(xaxis_title="오경보 비율 (충분한 날을 '부족'이라 한 비율)",
                      yaxis_title="재현율 (부족한 날을 잡아낸 비율)")
st.plotly_chart(fig_roc, use_container_width=True)
j = res["tpr"] - res["fpr"]
j_idx = int(np.argmax(j))
with st.expander("📘 ROC 곡선은 어떻게 읽어요?", expanded=True):
    st.markdown(f"""
- 가로축 **오경보 비율**(낮을수록 좋음), 세로축 **재현율**(높을수록 좋음).
- '몇 % 이상이면 부족으로 볼까?'(임계값)를 바꿀 때마다 점이 생겨요. 기준을 낮추면 많이 잡지만 오경보도 늘어요(**트레이드오프**).
- 점선은 동전 던지기예요. 곡선이 붙어 있으면 구분 능력이 없고, 왼쪽 위로 볼록할수록 좋아요. 면적이 AUC(**{auc:.3f}**)예요.
- 대각선에서 가장 먼 지점: 오경보 {res['fpr'][j_idx]:.1%}일 때 재현율 {res['tpr'][j_idx]:.1%}(차이 {j[j_idx]:.2f}). 0.2 미만이면 어떤 기준도 효과가 작아요.
- 곡선의 점은 {len(res['fpr'])}개. 적으면 시험 데이터가 적거나 점수가 몇 값에 몰려 있다는 뜻이에요.
- 확률 점수 표준편차 {res['proba_std']:.3f}: 매우 작으면 모든 날에 비슷한 점수를 줘서 구분을 못 해요.
""")

# ---------------- 3. 혼동행렬 ----------------
st.header("3️⃣ 혼동행렬")
cm = res["cm"]
fig_cm = px.imshow(cm, text_auto=True, x=["충분(예측)", "부족(예측)"],
                   y=["충분(실제)", "부족(실제)"], color_continuous_scale="Blues")
col_a, col_b = st.columns(2)
col_a.plotly_chart(fig_cm, use_container_width=True)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) else 0.0
lift = (tp / (tp + fp)) / res["test_prev"] if (tp + fp) and res["test_prev"] > 0 else 0.0
pred_pos_rate = (tp + fp) / cm.sum()
with col_b:
    st.markdown(f"""
**시험 {cm.sum()}건 (기준 0.5)**

| 칸 | 의미 | 개수 |
|---|---|---|
| 충분→충분 (TN) | 잘 맞힘 | **{tn}** |
| 부족→부족 (TP) | 잘 맞힘 | **{tp}** |
| 충분→부족 (FP) | 오경보 | **{fp}** |
| 부족→충분 (FN) | 놓침 | **{fn}** |
""")
if pred_pos_rate > res["test_prev"] + 0.15:
    pred_msg = "실제보다 훨씬 자주 '부족'이라 경고해요(balanced 영향이거나 근거가 약함)."
elif pred_pos_rate > res["test_prev"] - 0.15:
    pred_msg = "'부족' 예측 빈도가 실제 비율과 비슷해요."
else:
    pred_msg = "실제보다 '부족'을 훨씬 드물게 예측해요. 많이 놓칠 가능성이 커요."
st.markdown(f"""
- **특이도 {specificity:.1%}**: 실제 '충분'을 맞힌 비율. 재현율({rf['재현율']:.1%})과 평균하면 균형정확도({rf['균형정확도']:.1%}).
- **'부족' 예측 {pred_pos_rate:.1%}** vs **실제 {res['test_prev']:.1%}**: {pred_msg}
- **Lift {lift:.2f}**: 경고한 날의 실제 부족 비율이 평균의 몇 배인지. 1.0=무작위, 1.5 이상이면 의미 있어요.
- **놓침 {fn}건 vs 오경보 {fp}건**: 놓침이 문제면 재현율, 오경보가 부담이면 정밀도를 우선해요.
- 시험 데이터가 {cm.sum()}건뿐이라 몇 건만 바뀌어도 비율이 크게 변해요.
""")

# ---------------- 4. 변수 중요도 ----------------
st.header("4️⃣ 어떤 변수가 예측에 쓰였나?")
imp = res["importance"]
c1, c2 = st.columns(2)
c1.plotly_chart(px.bar(imp.sort_values("트리 중요도"), x="트리 중요도", y="변수",
                       orientation="h", title="트리 중요도 (학습 데이터)"),
                use_container_width=True)
c2.plotly_chart(px.bar(imp.sort_values("순열 중요도"), x="순열 중요도", y="변수",
                       orientation="h", error_x="순열 표준편차",
                       title="순열 중요도 (시험 데이터)"), use_container_width=True)
top_perm = imp.sort_values("순열 중요도", ascending=False).iloc[0]
top_tree = imp.sort_values("트리 중요도", ascending=False).iloc[0]
n_useful = int((imp["순열 중요도"] > imp["순열 표준편차"]).sum())
n_negative = int((imp["순열 중요도"] < 0).sum())
top_msg = ("두 방법의 1위가 **일치**해 일관적이에요." if top_perm["변수"] == top_tree["변수"]
           else "두 방법의 1위가 **달라요**. 학습에서 많이 쓰인 변수가 새 사용자에겐 도움이 안 됐을 수 있어요.")
sig_msg = ("1위 변수도 오차막대가 0을 포함해 의미 있는 차이로 보기 어려워요."
           if top_perm["순열 중요도"] <= top_perm["순열 표준편차"]
           else "1위 변수는 감소량이 표준편차보다 커서 어느 정도 기여했어요.")
with st.expander("📘 두 그래프는 무슨 차이예요?", expanded=True):
    st.markdown(f"""
- **트리 중요도**: 학습 중 그 변수로 얼마나 잘 나눴는지(합=1, 평균 0.2). 학습 기준이라 과대평가되기 쉽고 연속형 변수에 유리해요.
- **순열 중요도**: 시험 데이터에서 변수를 섞었을 때 AUC가 떨어지는 정도. **0 근처이거나 오차막대가 0을 지나면 도움이 안 된 변수**, 음수는 노이즈 신호예요.
- 트리 1위 **{top_tree['변수']}**({top_tree['트리 중요도']:.3f}), 순열 1위 **{top_perm['변수']}**(감소량 {top_perm['순열 중요도']:.3f}, 표준편차 {top_perm['순열 표준편차']:.3f}). {top_msg}
- {sig_msg}
- 5개 중 감소량이 표준편차보다 큰 변수 **{n_useful}개**, 음수 변수 **{n_negative}개**.
- 전체 AUC가 0.5 근처면 순위는 우연일 가능성이 커서 해석하지 않는 게 안전해요.
- ⚠️ 걸음 수와 활동 시간은 서로 연결돼 있어 하나를 섞어도 다른 변수가 대신해 중요도가 작게 나올 수 있어요.
- ⚠️ 중요도는 '모델이 의존한 정도'이지 '수면에 미치는 영향'이 아니에요. 방향은 6번 그래프를 보세요.
""")

# ---------------- 5. 모델 설정 ----------------
st.header("5️⃣ 모델 설정(하이퍼파라미터)")
c1, c2, c3 = st.columns(3)
c1.metric("트리 수", bp["n_estimators"])
c2.metric("최대 깊이", bp["max_depth"])
c3.metric("리프 최소 표본", bp["min_samples_leaf"])
st.caption(f"현재 설정 방식: **{mode}**")

spread_msg = ""
n_combo = 0
if tuned is not None:
    cvt = tuned["cv_table"]
    n_combo = len(cvt)
    top = cvt.iloc[0]
    worst = cvt.iloc[-1]
    spread = top["교차검증 AUC 평균"] - worst["교차검증 AUC 평균"]
    st.caption(f"교차검증 최고 AUC {top['교차검증 AUC 평균']:.3f}(표준편차 {top['표준편차']:.3f}) / "
               f"최저 {worst['교차검증 AUC 평균']:.3f} / 차이 {spread:.3f}")
    with st.expander("교차검증 전체 결과 보기"):
        st.dataframe(cvt.round(3), use_container_width=True)
    spread_msg = ("설정을 바꿔도 AUC가 거의 안 변해요. 설정 문제가 아니라 **입력 변수에 신호가 부족**하다는 뜻일 수 있어요."
                  if spread < 0.05 else "설정에 따라 점수 차이가 있어요. 튜닝이 어느 정도 의미 있어요.")
    if chosen != best:
        res_best = train(threshold, min_steps, test_size, best["n_estimators"],
                         best["max_depth"], best["min_samples_leaf"])
        if res_best is not None:
            best_auc = res_best["scores"].loc["랜덤 포레스트", "AUC"]
            st.subheader("내 설정 vs 최적 설정")
            cmp_df = pd.DataFrame({
                "내 설정": [chosen["n_estimators"], chosen["max_depth"],
                          chosen["min_samples_leaf"], res["train_auc"], res["cv_mean"], auc],
                "최적 설정": [best["n_estimators"], best["max_depth"],
                           best["min_samples_leaf"], res_best["train_auc"],
                           res_best["cv_mean"], best_auc],
            }, index=["트리 수", "최대 깊이", "리프 최소 표본",
                      "학습 AUC", "교차검증 AUC", "시험 AUC"])
            st.dataframe(cmp_df.round(3), use_container_width=True)
            st.markdown(f"""
- 시험 AUC는 내 설정이 최적보다 **{auc - best_auc:+.3f}**.
- 내 설정의 시험 AUC가 더 높아도 시험 사용자가 적어 **우연일 수 있어요.** 설정을 시험 점수로 고르면 점수가 부풀려져요.
- 내 설정의 학습 AUC만 훨씬 높다면 모델이 **외우고 있다**는 신호예요.
""")

depth_msg = ("얕은 트리라 데이터가 적을 때 유리해요." if bp["max_depth"] <= 3
             else "비교적 깊은 트리예요. 과적합 진단을 확인하세요.")
with st.expander("📘 이 설정들은 무슨 뜻이에요?", expanded=True):
    st.markdown(f"""
- **트리 수({bp['n_estimators']})**: 투표하는 트리 개수. 많을수록 안정적이지만 일정 수준 이상은 효과가 거의 없어요.
- **최대 깊이({bp['max_depth']})**: 질문 횟수. 깊을수록 외우기(과적합) 쉬워요. {depth_msg}
- **리프 최소 표본({bp['min_samples_leaf']})**: 마지막 가지의 최소 데이터 수. 클수록 단순해져요.
- **튜닝**: 학습 데이터 안의 사용자 단위 교차검증으로 골랐고, 시험 데이터는 마지막 평가에만 썼어요.
- **후보**: 트리 수 {GRID['n_estimators']}, 깊이 {GRID['max_depth']}, 리프 {GRID['min_samples_leaf']} (총 {n_combo}가지). 밖의 값은 '직접 설정'으로 실험하세요.
- {spread_msg}
- **실험 팁**: 깊이를 12까지 올리면 학습 AUC는 오르는데 시험 AUC는 떨어지는지 관찰해 보세요.
""")

# ---------------- 6. 직접 예측 + 민감도 ----------------
st.header("6️⃣ 직접 입력해서 예측해 보기")
cols = st.columns(5)
steps = cols[0].number_input("걸음 수", 0, 40000, 8000, 500)
very = cols[1].number_input("매우 활발(분)", 0, 300, 20)
fairly = cols[2].number_input("적당히 활발(분)", 0, 300, 20)
lightly = cols[3].number_input("가벼운 활동(분)", 0, 600, 200)
sed = cols[4].number_input("좌식(분)", 0, 1440, 800)
user_vals = [steps, very, fairly, lightly, sed]

p = float(res["model"].predict_proba(np.array([user_vals]))[0, 1])
st.metric("다음 날 수면 부족 확률(모델 점수)", f"{p:.1%}")
st.progress(min(max(p, 0.0), 1.0))
if abs(p - prev) < 0.05:
    p_msg = "평균 비율과 거의 같아서 특별한 단서를 못 찾았어요."
elif p > prev:
    p_msg = f"평균({prev:.1%})보다 **{(p - prev)*100:.1f}%p 높아요**."
else:
    p_msg = f"평균({prev:.1%})보다 **{(prev - p)*100:.1f}%p 낮아요**."
st.markdown(f"- {p_msg} AUC가 0.5 근처라면 이 값은 믿으면 안 돼요.")

st.subheader("민감도 분석: 한 변수만 바꾸면 확률이 어떻게 변할까?")
sel = st.selectbox("바꿔 볼 변수", FEATURES, format_func=lambda f: FEATURE_KR[f])
si = FEATURES.index(sel)
lo = float(data[sel].quantile(0.01))
hi = float(data[sel].quantile(0.99))
grid_x = np.linspace(lo, hi, 40)
rows = np.tile(np.array(user_vals, dtype=float), (len(grid_x), 1))
rows[:, si] = grid_x
probs = res["model"].predict_proba(rows)[:, 1]
fig_s = go.Figure()
fig_s.add_trace(go.Scatter(x=grid_x, y=probs, mode="lines+markers", name="예측 확률"))
fig_s.add_hline(y=prev, line_dash="dash", line_color="gray",
                annotation_text=f"평균 부족 비율 {prev:.1%}")
fig_s.update_layout(xaxis_title=FEATURE_KR[sel], yaxis_title="수면 부족 확률",
                    yaxis=dict(range=[0, 1]))
st.plotly_chart(fig_s, use_container_width=True)
rng = float(probs.max() - probs.min())
trend = float(np.corrcoef(grid_x, probs)[0, 1]) if probs.std() > 0 else 0.0
if rng < 0.05:
    s_msg = "확률이 거의 변하지 않아요. 모델이 이 변수에서 단서를 못 찾았다는 증거예요."
elif trend > 0.5:
    s_msg = "값이 커질수록 부족 확률이 **대체로 올라가요**."
elif trend < -0.5:
    s_msg = "값이 커질수록 부족 확률이 **대체로 내려가요**."
else:
    s_msg = "오르내림이 섞인 **비단조(U자형·계단형)** 패턴이에요. 우연일 수도 있어요."
st.markdown(f"""
- 범위 {lo:.0f}~{hi:.0f}에서 확률 변화폭은 **{rng*100:.1f}%p**예요. {s_msg}
- 랜덤 포레스트는 계단처럼 변하는 게 정상이에요(트리가 구간으로 나누기 때문).
- ⚠️ 다른 변수는 입력값에 고정했어요. 현실에 없는 조합일 수 있어요.
- ⚠️ 이 그래프는 **모델이 배운 패턴**이지 **인과관계**가 아니에요.
""")

# ---------------- 7. 종합 결론 ----------------
st.header("7️⃣ 종합 결론")
if auc < 0.55:
    st.error(f"시험 AUC **{auc:.3f}**로 동전 던지기 수준이에요. 이 데이터와 변수로는 다음 날 수면 부족을 예측할 신호를 찾지 못했어요.")
elif auc < 0.65:
    st.warning(f"시험 AUC **{auc:.3f}**로 약한 신호만 보여요. 우연일 수 있으니 기준을 바꿔 일관적인지 확인하세요.")
else:
    st.success(f"시험 AUC **{auc:.3f}**로 어느 정도 구분 능력이 있어요. 사용자 수가 적으니 다른 분할·기준에서도 같은지 확인하세요.")

cv_txt = "계산 불가" if np.isnan(res["cv_mean"]) else f"{res['cv_mean']:.3f}"
st.markdown(f"""
**현재 결과 요약**
- 시험 AUC **{auc:.3f}**, 교차검증 AUC **{cv_txt}**, 학습 AUC **{res['train_auc']:.3f}** (차이 {gap:+.3f})
- 균형정확도 **{rf['균형정확도']:.3f}**, Lift **{lift:.2f}**
- 순열 중요도에서 의미 있는 변수 **{n_useful}개** / 5개
- 설정: 트리 {bp['n_estimators']}개, 깊이 {bp['max_depth']}, 리프 {bp['min_samples_leaf']} ({mode})

**⚠️ 결론을 쓸 때 주의할 점**
1. "관계가 없다"가 아니라 **"이 데이터·변수로는 신호를 못 찾았다"**라고 써야 정확해요. 사용자가 {data['Id'].nunique()}명뿐이라 약한 관계는 못 잡을 수 있어요.
2. **하루 합계만** 사용했어요. 운동 시각 등은 몰라요.
3. **좌식 시간에 수면이 섞였을 수** 있어요.
4. 시험 사용자가 {res['test_users']}명뿐이라 분할이 바뀌면 점수가 크게 달라져요. 시험 비율을 바꿔 가며 확인하세요.
5. 사이드바에서 **수면 부족 기준을 6·7·8시간으로 바꿔** 결론이 그대로인지 확인하세요.
""")
st.caption("끝")
