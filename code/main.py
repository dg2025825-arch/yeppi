import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score

st.set_page_config(page_title="활동 패턴과 수면 부족", page_icon="😴", layout="wide")

ACT_NAME = "dailyActivity_merged.csv"
SLEEP_NAME = "sleepDay_merged.csv"

FEATURES = {
    "TotalSteps": "총 걸음 수",
    "VeryActiveMinutes": "고강도 활동(분)",
    "FairlyActiveMinutes": "중강도 활동(분)",
    "LightlyActiveMinutes": "저강도 활동(분)",
    "SedentaryMinutes": "좌식 시간(분)",
    "ActiveTotal": "활동 시간 합계(고+중강도, 분)",
}
BASE_FEATS = ["TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes",
              "LightlyActiveMinutes", "SedentaryMinutes"]
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
COLOR_MAP = {"수면 부족": "#EF553B", "충분한 수면": "#636EFA"}

# ------------------------------------------------------------------
# 1. 데이터 불러오기
# ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def find_file(keyword):
    for d in [REPO_ROOT / "data", REPO_ROOT, HERE, Path.cwd()]:
        if not d.exists():
            continue
        for p in d.rglob("*.csv"):
            if keyword.lower() in p.name.lower():
                return p
    return None


@st.cache_data
def load_data(act_src, sleep_src):
    act = pd.read_csv(act_src)
    sleep = pd.read_csv(sleep_src)

    act["ActivityDate"] = pd.to_datetime(act["ActivityDate"], format="%m/%d/%Y")
    sleep["SleepDay"] = pd.to_datetime(
        sleep["SleepDay"], format="%m/%d/%Y %I:%M:%S %p"
    ).dt.normalize()

    act["ActiveTotal"] = act["VeryActiveMinutes"] + act["FairlyActiveMinutes"]
    act["NextDay"] = act["ActivityDate"] + pd.Timedelta(days=1)

    # 오늘 활동 -> 내일 수면
    df = act.merge(sleep, left_on=["Id", "NextDay"],
                   right_on=["Id", "SleepDay"], how="inner")

    # 대조용: 활동일 당일 날짜의 수면 기록(= 활동 전날 밤 수면)
    prev = sleep[["Id", "SleepDay", "TotalMinutesAsleep"]].rename(
        columns={"SleepDay": "ActivityDate", "TotalMinutesAsleep": "어젯밤수면"})
    df = df.merge(prev, on=["Id", "ActivityDate"], how="left")

    df["요일"] = df["ActivityDate"].dt.dayofweek.map(lambda i: WEEKDAYS[i])
    df["Id"] = df["Id"].astype(str)
    return df


def get_data():
    act_path = find_file("dailyactivity")
    sleep_path = find_file("sleepday")
    if act_path is not None and sleep_path is not None:
        return (load_data(str(act_path), str(sleep_path)),
                f"{act_path.parent.name}/ 폴더 ({act_path.name}, {sleep_path.name})")

    st.error("저장소에서 CSV 파일을 찾지 못했습니다. 아래 정보를 확인하거나 직접 올려 주세요.")
    with st.expander("🔎 앱이 보고 있는 폴더와 파일 목록", expanded=True):
        st.write("main.py 위치:", str(HERE))
        files = sorted(str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob("*")
                       if p.is_file() and ".git" not in p.parts)
        st.write(files if files else "파일이 없습니다.")
    up_act = st.file_uploader(f"{ACT_NAME} 업로드", type="csv")
    up_sleep = st.file_uploader(f"{SLEEP_NAME} 업로드", type="csv")
    if up_act is None or up_sleep is None:
        st.stop()
    return load_data(up_act, up_sleep), "직접 업로드"


raw, source = get_data()

# ------------------------------------------------------------------
# 2. 사이드바 설정과 데이터 정리
# ------------------------------------------------------------------
st.sidebar.header("⚙️ 분석 설정")
st.sidebar.caption(f"📂 데이터 출처: {source}")

threshold = st.sidebar.slider(
    "수면 부족 기준 (분 미만)", 300, 480, 360, 10,
    help="360분 = 6시간, 420분 = 7시간. 기준에 따라 결과가 크게 달라질 수 있습니다.")
st.sidebar.caption(f"기준: {threshold // 60}시간 {threshold % 60}분 미만")

remove_noise = st.sidebar.checkbox("기기를 거의 안 찬 날 제외 (걸음 수 < 500)", value=True)
drop_short = st.sidebar.checkbox(
    "180분 미만 수면 기록 제외 (낮잠·부분 기록 의심)", value=False)

df = raw.copy()
if remove_noise:
    df = df[df["TotalSteps"] >= 500]
if drop_short:
    df = df[df["TotalMinutesAsleep"] >= 180]
df = df.reset_index(drop=True)

df["수면부족"] = (df["TotalMinutesAsleep"] < threshold).astype(int)
df["그룹"] = df["수면부족"].map({1: "수면 부족", 0: "충분한 수면"})

# 개인 평균 대비 편차 (개인 내 변화를 보기 위한 변수)
for f in FEATURES:
    df[f + "_dev"] = df[f] - df.groupby("Id")[f].transform("mean")
df["수면_dev"] = df["TotalMinutesAsleep"] - df.groupby("Id")["TotalMinutesAsleep"].transform("mean")

st.sidebar.markdown("---")
st.sidebar.metric("분석에 사용된 (활동일→다음날 수면) 쌍", len(df))
st.sidebar.metric("참가자 수", df["Id"].nunique())
st.sidebar.metric("수면 부족 비율", f"{df['수면부족'].mean() * 100:.1f}%")

if len(df) < 20 or df["Id"].nunique() < 5:
    st.error("선택한 조건으로 남은 데이터가 너무 적습니다. 사이드바 옵션을 조정하세요.")
    st.stop()


# ------------------------------------------------------------------
# 3. 계산 도구
# ------------------------------------------------------------------
def cv_auc(d, feats, y_col="수면부족", shuffle_seed=None):
    """같은 사람이 학습/검증에 섞이지 않도록 GroupKFold로 AUC 계산."""
    d = d.dropna(subset=feats)
    y = d[y_col].values
    if len(np.unique(y)) < 2 or d["Id"].nunique() < 3:
        return np.nan
    if shuffle_seed is not None:
        y = np.random.default_rng(shuffle_seed).permutation(y)
        if len(np.unique(y)) < 2:
            return np.nan
    n_split = min(5, d["Id"].nunique())
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000))
    proba = cross_val_predict(model, d[feats], y, cv=GroupKFold(n_splits=n_split),
                              groups=d["Id"], method="predict_proba")[:, 1]
    return roc_auc_score(y, proba)


def within_perm_p(x_dev, y_dev, groups_idx, n_perm=300, seed=0):
    """개인 내 상관의 순열검정: 사람 안에서만 수면 편차를 섞는다."""
    if np.std(x_dev) == 0 or np.std(y_dev) == 0:
        return np.nan, np.nan
    r = np.corrcoef(x_dev, y_dev)[0, 1]
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        yp = y_dev.copy()
        for g in groups_idx:
            yp[g] = rng.permutation(y_dev[g])
        if abs(np.corrcoef(x_dev, yp)[0, 1]) >= abs(r):
            cnt += 1
    return r, (cnt + 1) / (n_perm + 1)


@st.cache_data
def feature_table(d):
    gidx = [np.where(d["Id"].values == i)[0] for i in d["Id"].unique()]
    yd = d["수면_dev"].values
    rows = []
    for f, name in FEATURES.items():
        x = d[f].values
        pooled = spearmanr(x, d["TotalMinutesAsleep"])[0]
        r_in, p_in = within_perm_p(d[f + "_dev"].values, yd, gidx)
        if d["수면부족"].nunique() == 2 and np.std(x) > 0:
            a = roc_auc_score(d["수면부족"], x)
            auc1 = max(a, 1 - a)
        else:
            auc1 = np.nan
        rows.append({
            "변수": name,
            "전체 상관(스피어만)": round(pooled, 3),
            "개인 내 상관(편차)": round(r_in, 3),
            "개인 내 p값(순열)": round(p_in, 3),
            "단일변수 AUC": round(auc1, 3),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 4. 그래프별 해설
# ------------------------------------------------------------------
GUIDE = {
    "0": ("⭐⭐⭐⭐⭐ 전체 평가",
          "변수별로 '전체 상관'과 '개인 내 상관'을 나란히 계산합니다. 전체 상관은 크지만 개인 내 상관이 작으면, 관계가 사람 간 차이에서 왔다는 신호입니다.",
          "변수를 여섯 개 동시에 검정하므로 우연히 p값이 작게 나올 수 있습니다(다중비교). 표본이 작아 값이 불안정합니다."),
    "1": ("⭐⭐ 보조",
          "수면 시간 분포와 기준선 위치를 보여 줍니다. 라벨(수면 부족)이 믿을 만한지 점검합니다.",
          "활동 변수가 없어서 질문에 직접 답하지 못합니다. 짧은 기록은 낮잠이나 측정 오류일 수 있습니다."),
    "2": ("⭐⭐⭐ 관계 탐색",
          "활동량과 다음 날 수면 시간의 전체 관계입니다. 색을 '참가자'로 바꾸면 같은 사람이 뭉쳐 있는지 볼 수 있습니다.",
          "사람 간 차이와 개인 내 변화가 섞여 있습니다. 같은 사람의 점들은 독립이 아니고, 이상치에 민감합니다."),
    "3": ("⭐⭐⭐ 그룹 비교",
          "수면 부족 그룹과 충분 그룹의 전날 활동 분포를 비교합니다. 상자가 많이 겹치면 구분이 어렵습니다.",
          "한 변수씩만 비교하고, 몇 명이 한 그룹을 대부분 차지하면 왜곡됩니다."),
    "4": ("⭐⭐⭐ 경향 확인",
          "활동량 구간별 수면 부족 비율입니다. 방향이 일정한지, 구간 수를 바꿔도 유지되는지 봅니다.",
          "구간별 표본이 적어 비율이 흔들립니다. 신뢰구간이 없습니다."),
    "5": ("⭐⭐⭐ 훑어보기",
          "변수 간, 그리고 변수와 수면 시간의 상관 지도입니다. 변수끼리 겹치는지도 확인합니다.",
          "직선 관계만 측정하고, 사람 간 차이가 섞여 있습니다."),
    "6": ("⭐ 다른 수준의 질문",
          "사람별 평균 활동 vs 평균 수면입니다. 질문은 개인 내 변화인데 이 그래프는 사람 간 차이를 봅니다.",
          "참가자 수가 적어 점 하나가 경향을 좌우합니다."),
    "7": ("⭐⭐ 사례 관찰",
          "한 사람의 활동(막대)과 다음 날 수면(선)을 겹쳐 봅니다.",
          "일반화할 수 없고, 축 눈금에 따라 착시가 생깁니다."),
    "8": ("⭐⭐⭐⭐ 예측 성능",
          "활동 변수 전체로 수면 부족을 예측하는 모델의 AUC, 정확도, 혼동행렬입니다.",
          "표본이 작아 성능이 불안정하고, 좌식 시간에 수면 정보가 섞였을 수 있습니다. 12번에서 비교하세요."),
    "9": ("⭐⭐⭐⭐ 핵심 점검",
          "각 변수의 전체 상관과 개인 내 상관을 막대로 비교합니다. 두 막대 차이가 크면 사람 간 차이가 관계를 만들었다는 뜻입니다.",
          "개인 내 상관도 표본이 작으면 불안정합니다."),
    "10": ("⭐⭐⭐⭐⭐ 질문에 가장 적합",
           "x: 평소보다 오늘 활동이 얼마나 많았나, y: 평소보다 다음 날 수면이 얼마나 길었나. '같은 사람이 평소와 다르게 활동한 날'의 효과만 남깁니다.",
           "개인 평균은 분석에 쓰인 모든 날로 계산하므로 완전한 미래 정보 없는 예측은 아닙니다. 직선 관계만 봅니다."),
    "11": ("⭐⭐⭐⭐ 일관성 확인",
           "사람마다 활동과 수면의 상관계수를 따로 계산해 분포를 봅니다. 대부분 같은 방향이면 일관된 패턴입니다.",
           "사람당 기록이 적으면 상관계수가 우연에 크게 흔들립니다(최소 일수 슬라이더로 조절)."),
    "12": ("⭐⭐⭐⭐⭐ 예측 대조",
           "활동만, 좌식 제외, 어젯밤 수면, 무작위(셔플)를 같은 방식으로 비교합니다. 활동 모델이 무작위보다 얼마나 높은지가 핵심입니다.",
           "무작위 AUC는 20번 셔플의 평균이며, 표본이 작아 0.5 근처에서 흔들립니다. 어젯밤 수면은 비교용 대조이지 '활동만'이라는 질문의 답이 아닙니다."),
    "13": ("⭐⭐ 숨은 변수 점검",
           "요일에 따라 활동과 수면 부족 비율이 달라지는지 봅니다.",
           "요일별 표본이 적습니다. 요일은 활동과 수면 둘 다에 영향을 줄 수 있는 제3의 변수입니다."),
}


def show_guide(key):
    fit, what, limit = GUIDE[key]
    st.markdown("---")
    st.markdown(f"### 📖 이 그래프 해설  \n**질문 적합도:** {fit}")
    t1, t2 = st.tabs(["📌 무엇을 보여주나", "⚠️ 한계"])
    t1.markdown(what)
    t2.markdown(limit)


# ------------------------------------------------------------------
# 5. 메인 화면
# ------------------------------------------------------------------
st.title("😴 하루의 활동 패턴만으로 다음 날 수면 부족을 예측할 수 있을까?")
st.write("**오늘의 활동량**과 **다음 날 수면**의 관계를 여러 그래프로 살펴봅니다. "
         "0번 '자동 평가'에서 어떤 그래프가 의미 있는 신호를 보이는지 먼저 확인해 보세요.")

chart_options = [
    "0. 자동 평가: 어떤 변수가 의미 있는 신호를 보이나",
    "1. 수면 시간 분포 (히스토그램)",
    "2. 활동 vs 다음 날 수면 (산점도 + 추세선)",
    "3. 수면 부족 그룹별 활동 비교 (박스플롯)",
    "4. 활동량 구간별 수면 부족 비율 (막대그래프)",
    "5. 변수 간 상관관계 (히트맵)",
    "6. 참가자별 비교 (평균 활동 vs 평균 수면)",
    "7. 개인별 시계열 (활동과 수면 함께 보기)",
    "8. 예측 모델 성능 (ROC 곡선 · 혼동행렬)",
    "9. 🆕 사람 간 vs 개인 내 상관 비교",
    "10. 🆕 개인 내 변화 산점도 (평소 대비 편차)",
    "11. 🆕 사람별 상관계수 분포",
    "12. 🆕 모델 비교 (활동 vs 어젯밤 수면 vs 무작위)",
    "13. 🆕 요일별 비교",
]
choice = st.selectbox("📊 보고 싶은 그래프를 선택하세요", chart_options)
key = choice.split(".")[0]
st.markdown("---")

# ---------------- 0. 자동 평가 ----------------
if key == "0":
    st.subheader("변수별 신호 요약 (현재 사이드바 설정 기준)")
    tbl = feature_table(df)
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.caption("전체 상관 = 사람이 섞인 상관. 개인 내 상관 = 각자 평소 대비 편차끼리의 상관. "
               "p값 = 사람 안에서 수면을 섞어 만든 우연 분포와 비교. 단일변수 AUC는 방향 무시(0.5~1).")

    cand = tbl[(tbl["개인 내 상관(편차)"].abs() >= 0.2) & (tbl["개인 내 p값(순열)"] < 0.05)]
    if len(cand):
        st.success("개인 내 신호 후보: " + ", ".join(cand["변수"]) +
                   "  (6개 변수를 동시에 검정했으므로 우연일 가능성도 따져 보세요)")
    else:
        st.warning("개인 내 상관이 |0.2| 이상이면서 p<0.05인 변수가 없습니다. "
                   "'평소 대비 활동 변화만으로는 뚜렷한 신호가 없다'는 결과일 수 있습니다.")

    big_gap = tbl[(tbl["전체 상관(스피어만)"].abs() - tbl["개인 내 상관(편차)"].abs()) >= 0.15]
    if len(big_gap):
        st.info("전체 상관이 개인 내 상관보다 훨씬 큰 변수: " + ", ".join(big_gap["변수"]) +
                " → 관계가 '사람 간 차이'에서 왔을 가능성이 있습니다.")
    st.info("💭 이 표에서 숫자가 작다는 것은 '실패'가 아니라 하나의 연구 결과입니다. "
            "설정(기준 360/420, 짧은 수면 제외)을 바꿔도 같은 결론인지 확인해 보세요.")

# ---------------- 1. 히스토그램 ----------------
elif key == "1":
    fig = px.histogram(df, x="TotalMinutesAsleep", nbins=30, color="그룹",
                       color_discrete_map=COLOR_MAP,
                       labels={"TotalMinutesAsleep": "수면 시간(분)"},
                       title="다음 날 수면 시간 분포")
    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                  annotation_text=f"기준 {threshold}분")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💭 100분 이하의 짧은 기록은 진짜 수면 부족일까요, 낮잠이나 측정 오류일까요?")

# ---------------- 2. 산점도 ----------------
elif key == "2":
    feat = st.selectbox("활동 변수 선택", list(FEATURES.keys()), format_func=lambda x: FEATURES[x])
    color_by = st.radio("점 색상 기준", ["수면 부족 여부", "참가자"], horizontal=True)
    fig = px.scatter(df, x=feat, y="TotalMinutesAsleep",
                     color="그룹" if color_by == "수면 부족 여부" else "Id",
                     color_discrete_map=COLOR_MAP,
                     labels={feat: FEATURES[feat], "TotalMinutesAsleep": "다음 날 수면 시간(분)"},
                     hover_data=["Id", "ActivityDate"],
                     title=f"{FEATURES[feat]} vs 다음 날 수면 시간")
    x, y = df[feat].values, df["TotalMinutesAsleep"].values
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                             name="추세선", line=dict(color="black", dash="dash")))
    fig.add_hline(y=threshold, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("상관계수 (r)", f"{np.corrcoef(x, y)[0, 1]:.3f}")
    st.info("💭 색을 '참가자'로 바꾸면 같은 사람이 한 덩어리로 모이나요? 그렇다면 이 r은 무엇을 반영할까요?")

# ---------------- 3. 박스플롯 ----------------
elif key == "3":
    feat = st.selectbox("활동 변수 선택", list(FEATURES.keys()), format_func=lambda x: FEATURES[x])
    fig = px.box(df, x="그룹", y=feat, color="그룹", points="all",
                 color_discrete_map=COLOR_MAP, labels={feat: FEATURES[feat]},
                 title=f"수면 부족 여부에 따른 전날 {FEATURES[feat]}")
    st.plotly_chart(fig, use_container_width=True)
    summary = df.groupby("그룹")[feat].agg(["count", "mean", "median"]).round(1)
    summary.columns = ["표본 수", "평균", "중앙값"]
    st.dataframe(summary)
    st.info("💭 두 그룹의 상자가 많이 겹친다면, 이 변수로 두 그룹을 구분할 수 있을까요?")

# ---------------- 4. 구간별 막대 ----------------
elif key == "4":
    feat = st.selectbox("구간으로 나눌 변수", list(FEATURES.keys()), format_func=lambda x: FEATURES[x])
    n_bins = st.slider("구간 수", 3, 8, 4)
    tmp = df.copy()
    tmp["구간"] = pd.qcut(tmp[feat], q=n_bins, duplicates="drop")
    rate = tmp.groupby("구간", observed=True).agg(
        수면부족비율=("수면부족", "mean"), 표본수=("수면부족", "size")).reset_index()
    rate["수면부족비율"] *= 100
    rate["구간"] = rate["구간"].astype(str)
    fig = px.bar(rate, x="구간", y="수면부족비율", text="표본수",
                 labels={"수면부족비율": "수면 부족 비율(%)", "구간": FEATURES[feat]},
                 title=f"{FEATURES[feat]} 구간별 수면 부족 비율 (막대 위 숫자 = 표본 수)")
    fig.add_hline(y=df["수면부족"].mean() * 100, line_dash="dash", annotation_text="전체 평균")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💭 구간 수를 바꿔도 같은 경향인가요? 표본 수가 적은 구간은 얼마나 믿을 수 있을까요?")

# ---------------- 5. 히트맵 ----------------
elif key == "5":
    cols = list(FEATURES.keys()) + ["TotalMinutesAsleep"]
    corr = df[cols].corr().round(2)
    names = {**FEATURES, "TotalMinutesAsleep": "다음 날 수면 시간"}
    corr = corr.rename(index=names, columns=names)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto",
                    title="활동 변수와 다음 날 수면 시간의 상관관계")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💭 마지막 행(수면 시간)의 값이 모두 작다면 어떤 결론을 내릴 수 있을까요?")

# ---------------- 6. 참가자별 비교 ----------------
elif key == "6":
    feat = st.selectbox("활동 변수 선택", list(FEATURES.keys()), format_func=lambda x: FEATURES[x])
    per = df.groupby("Id").agg(평균활동=(feat, "mean"), 평균수면=("TotalMinutesAsleep", "mean"),
                              일수=("Id", "size")).reset_index()
    fig = px.scatter(per, x="평균활동", y="평균수면", size="일수", hover_name="Id",
                     labels={"평균활동": f"평균 {FEATURES[feat]}", "평균수면": "평균 수면 시간(분)"},
                     title="참가자별 평균 활동량 vs 평균 수면 시간 (점 크기 = 기록 일수)")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💭 '사람들 사이의 차이'와 '같은 사람의 날마다의 차이'는 다른 질문입니다. 우리가 알고 싶은 것은?")

# ---------------- 7. 개인별 시계열 ----------------
elif key == "7":
    counts = df["Id"].value_counts()
    pid = st.selectbox("참가자 선택 (기록 많은 순)", counts.index.tolist(),
                       format_func=lambda x: f"{x} ({counts[x]}일)")
    feat = st.selectbox("활동 변수 선택", list(FEATURES.keys()), format_func=lambda x: FEATURES[x])
    p = df[df["Id"] == pid].sort_values("ActivityDate")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=p["ActivityDate"], y=p[feat], name=FEATURES[feat], opacity=0.5))
    fig.add_trace(go.Scatter(x=p["ActivityDate"], y=p["TotalMinutesAsleep"],
                             name="다음 날 수면(분)", mode="lines+markers",
                             line=dict(color="crimson"), yaxis="y2"))
    fig.update_layout(title=f"참가자 {pid}: 활동량(막대)과 그 다음 날 수면(선)",
                      yaxis=dict(title=FEATURES[feat]),
                      yaxis2=dict(title="수면 시간(분)", overlaying="y", side="right"),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.info("💭 활동량이 많은 날 다음에 수면이 항상 늘거나 줄어드나요? 참가자마다 패턴이 같은가요?")

# ---------------- 8. 예측 모델 ----------------
elif key == "8":
    st.subheader("로지스틱 회귀로 '내일 수면 부족'을 예측해 보기")
    sel = st.multiselect("모델에 넣을 활동 변수", list(FEATURES.keys()),
                         default=["TotalSteps", "VeryActiveMinutes", "LightlyActiveMinutes",
                                  "SedentaryMinutes"],
                         format_func=lambda x: FEATURES[x])
    if len(sel) == 0:
        st.warning("변수를 1개 이상 선택하세요.")
    elif df["수면부족"].nunique() < 2:
        st.warning("수면 부족/충분 두 그룹이 모두 있어야 합니다. 기준을 조정해 보세요.")
    else:
        X, y, groups = df[sel], df["수면부족"], df["Id"]
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(class_weight="balanced", max_iter=1000))
        proba = cross_val_predict(model, X, y, cv=GroupKFold(n_splits=min(5, groups.nunique())),
                                  groups=groups, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc, acc = roc_auc_score(y, proba), accuracy_score(y, pred)
        baseline = max(y.mean(), 1 - y.mean())

        c1, c2, c3 = st.columns(3)
        c1.metric("AUC", f"{auc:.3f}", help="0.5 = 동전 던지기 수준")
        c2.metric("정확도", f"{acc * 100:.1f}%")
        c3.metric("기준선 정확도", f"{baseline * 100:.1f}%")

        left, right = st.columns(2)
        with left:
            fpr, tpr, _ = roc_curve(y, proba)
            roc = go.Figure()
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"모델 (AUC={auc:.2f})"))
            roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="무작위",
                                     line=dict(dash="dash")))
            roc.update_layout(title="ROC 곡선", xaxis_title="거짓 양성 비율",
                              yaxis_title="참 양성 비율")
            st.plotly_chart(roc, use_container_width=True)
        with right:
            fig_cm = px.imshow(confusion_matrix(y, pred), text_auto=True,
                               color_continuous_scale="Blues",
                               x=["예측: 충분", "예측: 부족"], y=["실제: 충분", "실제: 부족"],
                               title="혼동행렬")
            st.plotly_chart(fig_cm, use_container_width=True)

        model.fit(X, y)
        coef = pd.DataFrame({"변수": [FEATURES[s] for s in sel],
                             "계수": model.named_steps["logisticregression"].coef_[0]}
                            ).sort_values("계수")
        st.plotly_chart(px.bar(coef, x="계수", y="변수", orientation="h",
                               title="변수별 계수 (양수 = 수면 부족 가능성 ↑)"),
                        use_container_width=True)
        st.info("💭 AUC가 0.5에 가깝다면 어떤 뜻일까요? 12번에서 좌식 제외, 무작위와 비교해 보세요.")

# ---------------- 9. 사람 간 vs 개인 내 ----------------
elif key == "9":
    tbl = feature_table(df)
    long = tbl.melt(id_vars="변수",
                    value_vars=["전체 상관(스피어만)", "개인 내 상관(편차)"],
                    var_name="종류", value_name="상관계수")
    fig = px.bar(long, x="변수", y="상관계수", color="종류", barmode="group",
                 title="활동 변수 vs 다음 날 수면 시간: 전체 상관과 개인 내 상관")
    fig.add_hline(y=0, line_color="black")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    st.info("💭 두 막대의 크기와 부호가 비슷한가요? 크게 다르다면 '사람마다 다른 수면 습관'이 "
            "관계를 만들었다는 뜻일 수 있습니다.")

# ---------------- 10. 개인 내 변화 산점도 ----------------
elif key == "10":
    feat = st.selectbox("활동 변수 선택", list(FEATURES.keys()), format_func=lambda x: FEATURES[x])
    x = df[feat + "_dev"].values
    y = df["수면_dev"].values
    fig = px.scatter(df, x=feat + "_dev", y="수면_dev", color="Id",
                     hover_data=["Id", "ActivityDate"],
                     labels={feat + "_dev": f"평소 대비 {FEATURES[feat]} 차이",
                             "수면_dev": "평소 대비 다음 날 수면 차이(분)"},
                     title=f"개인 내 변화: {FEATURES[feat]}")
    if np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                 name="추세선", line=dict(color="black", dash="dash")))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    gidx = [np.where(df["Id"].values == i)[0] for i in df["Id"].unique()]
    r, p = within_perm_p(x, y, gidx)
    c1, c2 = st.columns(2)
    c1.metric("개인 내 상관 (r)", f"{r:.3f}")
    c2.metric("순열검정 p값", f"{p:.3f}")
    st.info("💭 오른쪽 위/왼쪽 아래로 향하는 띠가 보이나요? 점이 원점 주위에 구름처럼 퍼져 있다면 "
            "'평소와 다른 활동'이 다음 날 수면을 설명하지 못한다는 뜻일 수 있습니다.")

# ---------------- 11. 사람별 상관계수 ----------------
elif key == "11":
    feat = st.selectbox("활동 변수 선택", list(FEATURES.keys()), format_func=lambda x: FEATURES[x])
    min_days = st.slider("사람당 최소 기록 일수", 5, 25, 10)
    rows = []
    for pid, g in df.groupby("Id"):
        if len(g) >= min_days and g[feat].std() > 0 and g["TotalMinutesAsleep"].std() > 0:
            rows.append({"Id": pid, "일수": len(g),
                         "상관계수": np.corrcoef(g[feat], g["TotalMinutesAsleep"])[0, 1]})
    res = pd.DataFrame(rows)
    if len(res) < 3:
        st.warning("조건을 만족하는 사람이 너무 적습니다. 최소 일수를 낮춰 보세요.")
    else:
        res = res.sort_values("상관계수")
        fig = px.bar(res, x="Id", y="상관계수", hover_data=["일수"],
                     title=f"사람별 상관계수: {FEATURES[feat]} vs 다음 날 수면 시간")
        fig.add_hline(y=0, line_color="black")
        st.plotly_chart(fig, use_container_width=True)
        pos = int((res["상관계수"] > 0).sum())
        neg = int((res["상관계수"] < 0).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("양의 상관 사람 수", pos)
        c2.metric("음의 상관 사람 수", neg)
        c3.metric("상관계수 중앙값", f"{res['상관계수'].median():.3f}")
    st.info("💭 대부분의 막대가 같은 방향인가요, 양쪽으로 갈리나요? 갈린다면 '모든 사람에게 통하는 규칙'이 있다고 말할 수 있을까요?")

# ---------------- 12. 모델 비교 ----------------
elif key == "12":
    st.subheader("같은 방식(GroupKFold)으로 비교한 AUC")
    if df["수면부족"].nunique() < 2:
        st.warning("수면 부족/충분 두 그룹이 모두 있어야 합니다.")
    else:
        with st.spinner("계산 중..."):
            no_sed = [f for f in BASE_FEATS if f != "SedentaryMinutes"]
            res = {
                "활동 전체 (좌식 포함)": cv_auc(df, BASE_FEATS),
                "활동 (좌식 제외)": cv_auc(df, no_sed),
                "걸음 수만": cv_auc(df, ["TotalSteps"]),
                "어젯밤 수면만 (대조)": cv_auc(df, ["어젯밤수면"]),
                "활동 + 어젯밤 수면 (대조)": cv_auc(df, BASE_FEATS + ["어젯밤수면"]),
            }
            shuffled = [cv_auc(df, BASE_FEATS, shuffle_seed=s) for s in range(20)]
            res["무작위 (정답 셔플 평균)"] = float(np.nanmean(shuffled))
        out = pd.DataFrame({"모델": list(res.keys()), "AUC": [round(v, 3) for v in res.values()]})
        fig = px.bar(out, x="모델", y="AUC", text="AUC", title="모델별 AUC (0.5 = 동전 던지기)")
        fig.add_hline(y=0.5, line_dash="dash", annotation_text="무작위 기준")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out, hide_index=True, use_container_width=True)
        n_nan = int(df["어젯밤수면"].isna().sum())
        st.caption(f"'어젯밤 수면' 모델은 어젯밤 기록이 없는 {n_nan}행을 제외하고 계산합니다.")
    st.info("💭 활동 모델의 AUC가 무작위와 얼마나 다른가요? 좌식을 빼면 떨어지나요? "
            "어젯밤 수면이 더 잘 맞춘다면, 활동만으로 예측한다는 질문에 대해 무엇을 말해 줄까요?")

# ---------------- 13. 요일별 비교 ----------------
elif key == "13":
    by = df.groupby("요일").agg(수면부족비율=("수면부족", "mean"),
                              평균걸음=("TotalSteps", "mean"),
                              표본수=("수면부족", "size")).reindex(WEEKDAYS).reset_index()
    by["수면부족비율"] = by["수면부족비율"] * 100
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(by, x="요일", y="수면부족비율", text="표본수",
                           title="활동한 요일별, 다음 날 수면 부족 비율(%)"),
                    use_container_width=True)
    c2.plotly_chart(px.bar(by, x="요일", y="평균걸음", title="요일별 평균 걸음 수"),
                    use_container_width=True)
    st.info("💭 요일에 따라 활동량도 수면 부족도 달라진다면, 둘의 관계는 '요일'이라는 "
            "제3의 변수 때문일 수 있습니다.")

# ---------------- 선택한 그래프의 해설 표시 ----------------
show_guide(key)

# ------------------------------------------------------------------
# 6. 전체 한계점
# ------------------------------------------------------------------
st.markdown("---")
with st.expander("⚠️ 이 분석 전체의 한계 (결론 내리기 전에 꼭 읽어보세요)"):
    st.markdown(
        """
- **표본이 작습니다.** 참가자 수가 적고, 수면 기록이 없는 날이 많아 분석에 쓰인 날 수가 제한적입니다.
- **선택 편향.** 수면 기록이 있는 날만 분석에 들어갑니다. 기록을 남기는 날과 안 남기는 날이 다를 수 있습니다.
- **수면 기록의 정의.** 낮잠이나 짧은 기록도 하루 수면으로 잡힐 수 있습니다.
- **기준선은 임의적입니다.** 360분이나 420분 같은 기준은 바꿀 수 있고, 기준에 따라 결과가 달라집니다.
- **상관관계 ≠ 인과관계.** 활동이 수면에 영향을 준 것인지, 다른 요인(시험, 스트레스, 생활 습관)이 있는지는 알 수 없습니다.
- **기기 착용 여부.** 기기를 안 찬 날은 활동이 낮게 기록됩니다. (사이드바에서 제외 가능)
- **참가자 차이.** 사람마다 기본 수면 패턴이 다르므로 개인차가 결과를 흐릴 수 있습니다.
- **좌식 시간의 누수 가능성.** 수면 기록이 있는 날은 좌식 시간에서 잠 시간이 빠져, 수면 정보가 섞였을 수 있습니다.
- **다중 비교.** 여러 변수와 그래프를 동시에 보면 우연히 의미 있어 보이는 결과가 나올 수 있습니다.
        """
    )
