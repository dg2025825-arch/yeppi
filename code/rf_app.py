import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, balanced_accuracy_score,
                             confusion_matrix, roc_curve)

# =========================================================
# 기본 설정
# =========================================================
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


# =========================================================
# 데이터 / 모델 함수
# =========================================================
@st.cache_data
def load_dataset(threshold, min_steps):
    """오늘 활동 + 내일 수면을 합치고, 수면 부족(1) 라벨을 만든다."""
    act = pd.read_csv(ACTIVITY_PATH)
    act["Id"] = act["Id"].astype(str)
    act["ActivityDate"] = pd.to_datetime(act["ActivityDate"], format="%m/%d/%Y")
    act["NextDate"] = act["ActivityDate"] + pd.Timedelta(days=1)

    sl = pd.read_csv(SLEEP_PATH)
    sl["Id"] = sl["Id"].astype(str)
    sl["SleepDate"] = pd.to_datetime(sl["SleepDay"], errors="coerce").dt.normalize()

    data = act.merge(
        sl[["Id", "SleepDate", "TotalMinutesAsleep"]],
        left_on=["Id", "NextDate"], right_on=["Id", "SleepDate"], how="inner",
    )
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


@st.cache_data(show_spinner="랜덤 포레스트를 학습하는 중입니다...")
def train(threshold, min_steps, test_size, tune):
    data = load_dataset(threshold, min_steps)
    if data["label"].nunique() < 2 or len(data) < 50 or data["Id"].nunique() < 6:
        return None

    X = data[FEATURES].values
    y = data["label"].values
    groups = data["Id"].values

    # 사용자 단위 분할 (같은 사람이 학습/시험에 섞이지 않음)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None

    base_rf = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
    cv_table = None
    if tune:
        grid = {"n_estimators": [100, 300],
                "max_depth": [2, 3, 4, 6],
                "min_samples_leaf": [5, 10, 20]}
        n_groups = len(np.unique(groups[tr]))
        gs = GridSearchCV(base_rf, grid, scoring="roc_auc",
                          cv=GroupKFold(n_splits=min(4, n_groups)), n_jobs=-1)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        model, best_params = gs.best_estimator_, gs.best_params_
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
    else:
        best_params = {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 10}
        model = RandomForestClassifier(class_weight="balanced", random_state=SEED,
                                       n_jobs=-1, **best_params).fit(X[tr], y[tr])

    dummy = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
    rf_proba = model.predict_proba(X[te])[:, 1]
    dm_proba = dummy.predict_proba(X[te])[:, 1]
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
        "data": data, "model": model, "best_params": best_params,
        "cv_table": cv_table, "scores": scores, "importance": importance,
        "cm": confusion_matrix(y[te], (rf_proba >= 0.5).astype(int)),
        "fpr": fpr, "tpr": tpr,
        "n_train": len(tr), "n_test": len(te),
        "train_users": len(np.unique(groups[tr])),
        "test_users": len(np.unique(groups[te])),
        "test_prev": float(y[te].mean()),
    }


# =========================================================
# 사이드바
# =========================================================
st.sidebar.header("⚙️ 설정")
threshold = st.sidebar.slider("수면 부족 기준(분)", 300, 540, 420, 30,
                              help="이 값보다 적게 자면 '수면 부족'. 420분 = 7시간")
min_steps = st.sidebar.slider("최소 걸음 수", 0, 3000, 500, 100,
                              help="기기를 거의 안 찬 날(걸음 수가 너무 적은 날)을 제외")
test_size = st.sidebar.slider("시험 데이터 비율(사용자 기준)", 0.2, 0.5, 0.3, 0.05)
tune = st.sidebar.checkbox("하이퍼파라미터 튜닝(교차검증)", value=True,
                           help="끄면 고정된 설정으로 빠르게 학습해요.")
st.sidebar.caption(f"난수 고정값: {SEED}")

# =========================================================
# 제목
# =========================================================
st.title("🌲 랜덤 포레스트로 다음 날 수면 부족 예측하기")
st.markdown("""
**연구 질문**: 하루의 활동 패턴(걸음 수, 활동 시간)만으로 **다음 날 수면 부족 여부**를 예측할 수 있는가?

> 랜덤 포레스트는 "걸음 수가 많은가?", "좌식 시간이 긴가?" 같은 질문으로 나뉘는 **결정 트리 여러 개가 투표**해서
> 결과를 정하는 모델이에요. 직선 관계뿐 아니라 U자형·조합 관계도 찾아낼 수 있어요.
""")

if not (os.path.exists(ACTIVITY_PATH) and os.path.exists(SLEEP_PATH)):
    st.error("data 폴더에서 dailyActivity_merged.csv 또는 sleepDay_merged.csv를 찾을 수 없어요. "
             f"확인한 경로: {os.path.normpath(DATA_DIR)}")
    st.stop()

res = train(threshold, min_steps, test_size, tune)
if res is None:
    st.warning("데이터가 너무 적거나 한쪽 라벨만 있어요. 사이드바의 기준을 조정해 보세요.")
    st.stop()

data = res["data"]
prev = data["label"].mean()
sc = res["scores"]
rf = sc.loc["랜덤 포레스트"]
dm = sc.loc["Dummy(다수 클래스)"]

# =========================================================
# 0. 데이터 요약
# =========================================================
st.header("0️⃣ 데이터 한눈에 보기")
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 데이터(건)", f"{len(data)}")
c2.metric("사용자 수(명)", f"{data['Id'].nunique()}")
c3.metric("수면 부족 비율", f"{prev:.1%}")
c4.metric("학습 / 시험(건)", f"{res['n_train']} / {res['n_test']}")

with st.expander("📘 이 숫자들은 무슨 뜻이에요?", expanded=True):
    st.markdown(f"""
- **전체 데이터**: '오늘 활동 → 내일 수면' 쌍이 {len(data)}개라는 뜻이에요. 활동 기록과 다음 날 수면 기록이 **둘 다 있는 날만** 포함돼요.
- **수면 부족 비율 {prev:.1%}**: 다음 날 {threshold}분({threshold/60:.1f}시간) 미만으로 잔 날의 비율이에요.
  한쪽이 너무 많으면 찍기만 해도 정확도가 높아 보이니 조심해야 해요.
- **학습 / 시험**: 모델이 공부한 데이터와 처음 보는 데이터예요. 사용자 단위로 나눴어요
  (학습 {res['train_users']}명, 시험 {res['test_users']}명, **서로 겹치지 않음**).
  같은 사람이 양쪽에 있으면 그 사람의 습관을 외워서 점수가 부풀려지기 때문이에요.
""")

# =========================================================
# 1. 성능 비교
# =========================================================
st.header("1️⃣ 시험 성적: 랜덤 포레스트 vs 찍기(Dummy)")

st.dataframe(
    sc.round(3).style.highlight_max(axis=0, color="#d4edda"),
    use_container_width=True,
)

fig_perf = go.Figure()
for name in sc.index:
    fig_perf.add_trace(go.Bar(name=name, x=sc.columns.tolist(), y=sc.loc[name].values))
fig_perf.update_layout(barmode="group", yaxis=dict(range=[0, 1]),
                       title="지표별 점수 비교 (높을수록 좋음)")
st.plotly_chart(fig_perf, use_container_width=True)

auc = rf["AUC"]
if auc < 0.55:
    verdict = "🔴 **동전 던지기 수준**이에요. 활동 패턴에서 수면 부족의 단서를 거의 찾지 못했어요."
elif auc < 0.65:
    verdict = "🟡 **약한 신호**가 있어요. 하지만 실제 예측에 쓰기엔 부족하고, 우연일 수도 있어요."
elif auc < 0.8:
    verdict = "🟢 **어느 정도 구분**해요. 다만 시험 사용자가 적어 반복 확인이 필요해요."
else:
    verdict = "🟢 **꽤 잘 구분**해요. 너무 좋게 나왔다면 데이터 누수나 우연을 먼저 의심해 보세요."

st.info(f"랜덤 포레스트 AUC **{auc:.3f}** → {verdict}")

with st.expander("📘 각 지표는 무슨 뜻이고, 우리 결과는 어때요?", expanded=True):
    st.markdown(f"""
(양성 = **수면 부족**. 아래 숫자는 현재 시험 데이터 기준이에요.)

| 지표 | 우리 데이터에서의 의미 | 랜덤 포레스트 | Dummy |
|---|---|---|---|
| **AUC** ⭐ | 부족인 날과 충분한 날을 하나씩 뽑았을 때 부족인 날에 더 높은 점수를 줄 확률. 0.5 = 동전 던지기 | **{rf['AUC']:.3f}** | {dm['AUC']:.3f} |
| **균형정확도** | 부족일 때 맞힌 비율과 충분일 때 맞힌 비율의 평균 | {rf['균형정확도']:.3f} | {dm['균형정확도']:.3f} |
| **F1** | 정밀도와 재현율의 조화평균 | {rf['F1']:.3f} | {dm['F1']:.3f} |
| **정밀도** | "내일 수면 부족"이라 경고한 날 중 실제 부족이었던 비율 | {rf['정밀도']:.3f} | {dm['정밀도']:.3f} |
| **재현율** | 실제 수면 부족인 날 중 잡아낸 비율 | {rf['재현율']:.3f} | {dm['재현율']:.3f} |
| **정확도** | 전체 중 맞힌 비율 | {rf['정확도']:.3f} | {dm['정확도']:.3f} |

**어떻게 읽나요?**
- 연구 질문이 "구분할 수 있는가"이므로 **AUC와 균형정확도를 가장 중요하게** 봐요. Dummy는 이 둘이 항상 0.5예요.
- **정확도는 조심**해서 보세요. 시험 데이터의 수면 부족 비율이 {res['test_prev']:.1%}라서, 아무 정보 없이 다수 쪽만 찍어도 약 {max(res['test_prev'], 1-res['test_prev']):.0%}가 나와요.
- **정밀도**가 수면 부족 기본 비율({res['test_prev']:.1%})과 비슷하면, 모델의 경고에 새로운 정보가 없다는 뜻이에요.
- **재현율만 높은 경우**는 "거의 다 부족이라고 찍어서" 생긴 값일 수 있어요. 정밀도와 같이 보세요.
- 랜덤 포레스트가 Dummy보다 **AUC·균형정확도가 0.5에서 얼마나 높은지**가 핵심이에요.
  지금은 AUC 차이가 **{auc - 0.5:+.3f}**, 균형정확도 차이가 **{rf['균형정확도'] - 0.5:+.3f}**예요.
- ⚠️ 시험 사용자가 {res['test_users']}명뿐이라, **다른 사용자가 시험에 들어가면 점수가 크게 달라질 수 있어요.**
""")

# =========================================================
# 2. ROC 곡선
# =========================================================
st.header("2️⃣ ROC 곡선: 구분 능력을 그림으로 보기")
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=res["fpr"], y=res["tpr"], mode="lines",
                             name=f"랜덤 포레스트 (AUC {auc:.3f})", line=dict(width=3)))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="찍기 (0.5)",
                             line=dict(dash="dash", color="gray")))
fig_roc.update_layout(xaxis_title="오경보 비율 (충분한 날을 '부족'이라 한 비율)",
                      yaxis_title="재현율 (부족한 날을 잡아낸 비율)")
st.plotly_chart(fig_roc, use_container_width=True)

with st.expander("📘 ROC 곡선은 어떻게 읽어요?", expanded=True):
    st.markdown("""
- **점선(대각선)**은 동전 던지기예요. 곡선이 여기에 붙어 있으면 구분 능력이 없다는 뜻이에요.
- 곡선이 **왼쪽 위로 볼록할수록** 오경보는 적고 잘 잡아내는, 좋은 모델이에요.
- 곡선 아래 면적이 **AUC**예요.
- 시험 데이터가 적으면 곡선이 계단처럼 들쭉날쭉해요. 한 번의 분할 결과이므로 너무 믿지 마세요.
""")

# =========================================================
# 3. 혼동행렬
# =========================================================
st.header("3️⃣ 혼동행렬: 어디서 맞고 어디서 틀렸나?")
cm = res["cm"]
fig_cm = px.imshow(cm, text_auto=True, x=["충분(예측)", "부족(예측)"],
                   y=["충분(실제)", "부족(실제)"], color_continuous_scale="Blues")
col_a, col_b = st.columns([1, 1])
col_a.plotly_chart(fig_cm, use_container_width=True)

tn, fp, fn, tp = cm.ravel()
with col_b:
    st.markdown(f"""
**시험 데이터 {cm.sum()}건의 결과 (기준 0.5)**

| 칸 | 의미 | 개수 |
|---|---|---|
| 충분 → 충분 | 잘 맞힘 | **{tn}** |
| 부족 → 부족 | 잘 맞힘 | **{tp}** |
| 충분 → 부족 | 오경보 (괜히 경고) | **{fp}** |
| 부족 → 충분 | 놓침 (경고 못 함) | **{fn}** |

- 대각선(왼쪽 위·오른쪽 아래)이 클수록 좋아요.
- 한 줄(예: 부족 예측)에만 몰려 있으면 모델이 거의 한쪽으로만 찍고 있다는 뜻이에요.
""")

# =========================================================
# 4. 변수 중요도
# =========================================================
st.header("4️⃣ 어떤 활동 변수가 예측에 쓰였나?")
imp = res["importance"]
c1, c2 = st.columns(2)
c1.plotly_chart(
    px.bar(imp.sort_values("트리 중요도"), x="트리 중요도", y="변수", orientation="h",
           title="트리 중요도 (학습 데이터 기준)"),
    use_container_width=True)
c2.plotly_chart(
    px.bar(imp.sort_values("순열 중요도"), x="순열 중요도", y="변수", orientation="h",
           error_x="순열 표준편차", title="순열 중요도 (시험 데이터 기준)"),
    use_container_width=True)

top_perm = imp.sort_values("순열 중요도", ascending=False).iloc[0]
with st.expander("📘 두 그래프는 무슨 차이예요?", expanded=True):
    st.markdown(f"""
- **트리 중요도**: 학습 중 트리가 그 변수로 얼마나 자주, 효과적으로 나눴는지예요. 학습 데이터에서 계산돼서 **과대평가되기 쉬워요.**
  (모든 변수 합 = 1이라 값이 있다고 해서 꼭 쓸모 있다는 뜻은 아니에요.)
- **순열 중요도**: 시험 데이터에서 그 변수의 값을 **무작위로 섞었을 때 AUC가 얼마나 떨어지는지**예요.
  많이 떨어지면 그 변수가 정말 도움이 됐다는 뜻이고, **0 근처이거나 오차막대가 0을 지나면 도움이 안 된 변수**예요.
- 우리 결과에서 순열 중요도 1위는 **{top_perm['변수']}** (AUC 감소량 {top_perm['순열 중요도']:.3f}, 표준편차 {top_perm['순열 표준편차']:.3f})예요.
  {"오차막대가 0을 포함하므로 의미 있는 차이라고 보기 어려워요." if top_perm['순열 중요도'] <= top_perm['순열 표준편차'] else "감소량이 표준편차보다 커서 어느 정도 기여한 것으로 볼 수 있어요."}
- 전체 AUC가 0.5 근처라면 중요도 순위는 **우연의 결과**일 가능성이 커서 해석하지 않는 편이 안전해요.
""")

# =========================================================
# 5. 튜닝 결과
# =========================================================
st.header("5️⃣ 모델 설정(하이퍼파라미터)")
bp = res["best_params"]
c1, c2, c3 = st.columns(3)
c1.metric("트리 수", bp["n_estimators"])
c2.metric("최대 깊이", bp["max_depth"])
c3.metric("리프 최소 표본", bp["min_samples_leaf"])

if res["cv_table"] is not None:
    top = res["cv_table"].iloc[0]
    st.caption(f"교차검증 최고 AUC {top['교차검증 AUC 평균']:.3f} (표준편차 {top['표준편차']:.3f})")
    with st.expander("교차검증 전체 결과 보기"):
        st.dataframe(res["cv_table"].round(3), use_container_width=True)

with st.expander("📘 이 설정들은 무슨 뜻이에요?", expanded=True):
    st.markdown("""
- **트리 수**: 투표에 참여하는 결정 트리 개수예요. 많을수록 안정적이지만 일정 수준 넘으면 효과가 거의 없어요.
- **최대 깊이**: 트리가 질문을 몇 번까지 던질 수 있는지예요. **깊을수록 학습 데이터를 외우기 쉬워서(과적합)** 우리처럼 데이터가 적으면 얕게 제한해요.
- **리프 최소 표본**: 마지막 가지에 최소 몇 건이 있어야 하는지예요. 클수록 모델이 단순해져요.
- **튜닝 방법**: 학습 데이터 안에서 **사용자 단위 교차검증**으로 가장 좋은 조합을 골랐고, **시험 데이터는 마지막 평가에만** 썼어요.
  시험 데이터로 설정을 고르면 점수가 부풀려지기 때문이에요.
- 교차검증 AUC도 0.5 근처라면, 설정을 바꿔도 소용이 없다 = 입력 변수에 신호가 부족하다는 뜻이에요.
""")

# =========================================================
# 6. 직접 예측
# =========================================================
st.header("6️⃣ 직접 입력해서 예측해 보기")
c = st.columns(5)
steps = c[0].number_input("걸음 수", 0, 40000, 8000, 500)
very = c[1].number_input("매우 활발(분)", 0, 300, 20)
fairly = c[2].number_input("적당히 활발(분)", 0, 300, 20)
lightly = c[3].number_input("가벼운 활동(분)", 0, 600, 200)
sed = c[4].number_input("좌식(분)", 0, 1440, 800)

p = float(res["model"].predict_proba(np.array([[steps, very, fairly, lightly, sed]]))[0, 1])
st.metric("다음 날 수면 부족 확률(모델 점수)", f"{p:.1%}")
st.progress(min(max(p, 0.0), 1.0))
st.caption("⚠️ 이 값은 모델의 점수일 뿐이에요. 위의 AUC가 0.5 근처라면 실제 예측으로 믿으면 안 돼요. "
           "입력값을 바꿔도 확률이 거의 안 변한다면, 모델이 활동 변수에서 단서를 못 찾았다는 증거예요.")

# =========================================================
# 7. 종합 결론
# =========================================================
st.header("7️⃣ 종합 결론")
if auc < 0.55:
    st.error(
        f"랜덤 포레스트의 시험 AUC는 **{auc:.3f}**로 동전 던지기 수준이에요. "
        "이 데이터와 변수로는 다음 날 수면 부족을 예측할 만한 신호를 찾지 못했어요.")
elif auc < 0.65:
    st.warning(
        f"랜덤 포레스트의 시험 AUC는 **{auc:.3f}**로 약한 신호만 보여요. "
        "우연의 범위일 수 있으니 기준을 바꿔서 결과가 일관적인지 확인해야 해요.")
else:
    st.success(
        f"랜덤 포레스트의 시험 AUC는 **{auc:.3f}**로 어느 정도 구분 능력이 있어요. "
        "다만 사용자 수가 적으니 다른 분할과 다른 기준에서도 같은지 반드시 확인하세요.")

st.markdown("""
**⚠️ 결론을 쓸 때 주의할 점**
1. **"관계가 없다"가 아니라 "이 데이터·변수로는 신호를 못 찾았다"**라고 써야 정확해요. 사용자가 약 24명뿐이라 약한 관계는 못 잡을 수 있어요.
2. **하루 합계만 사용**했어요. 운동 시각 등은 알 수 없어요.
3. **좌식 시간에 수면이 섞였을 수 있어요.** 좌식 변수의 해석이 흐려질 수 있어요.
4. **시험 사용자가 적어서** 분할이 바뀌면 점수가 크게 달라져요. 메인의 반복 분할 방식으로도 확인하면 더 탄탄해요.
5. 사이드바에서 **수면 부족 기준을 6시간·7시간·8시간으로 바꿔** 결론이 그대로인지 확인해 보세요.
""")
