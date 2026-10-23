import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score

st.set_page_config(page_title="활동 패턴과 수면 부족", page_icon="😴", layout="wide")

# ------------------------------------------------------------------
# 1. 데이터 불러오기 및 전처리
# ------------------------------------------------------------------
FEATURES = {
    "TotalSteps": "총 걸음 수",
    "VeryActiveMinutes": "고강도 활동(분)",
    "FairlyActiveMinutes": "중강도 활동(분)",
    "LightlyActiveMinutes": "저강도 활동(분)",
    "SedentaryMinutes": "좌식 시간(분)",
    "ActiveTotal": "활동 시간 합계(고+중강도, 분)",
}


@st.cache_data
def load_data(act_file, sleep_file):
    act = pd.read_csv(act_file)
    sleep = pd.read_csv(sleep_file)

    act["ActivityDate"] = pd.to_datetime(act["ActivityDate"], format="%m/%d/%Y")
    sleep["SleepDay"] = pd.to_datetime(
        sleep["SleepDay"], format="%m/%d/%Y %I:%M:%S %p"
    ).dt.normalize()

    act["ActiveTotal"] = act["VeryActiveMinutes"] + act["FairlyActiveMinutes"]

    # 핵심: '오늘의 활동' → '내일의 수면' 연결
    act["NextDay"] = act["ActivityDate"] + pd.Timedelta(days=1)
    df = act.merge(
        sleep,
        left_on=["Id", "NextDay"],
        right_on=["Id", "SleepDay"],
        how="inner",
    )
    df["Id"] = df["Id"].astype(str)
    return df


def find_file(name):
    base = Path(__file__).parent
    for p in [base / name, base / "data" / name]:
        if p.exists():
            return p
    return None


act_path = find_file("dailyActivity_merged.csv")
sleep_path = find_file("sleepDay_merged.csv")

if act_path is None or sleep_path is None:
    st.warning("CSV 파일을 찾지 못했습니다. 직접 업로드해 주세요.")
    up_act = st.file_uploader("dailyActivity_merged.csv", type="csv")
    up_sleep = st.file_uploader("sleepDay_merged.csv", type="csv")
    if up_act is None or up_sleep is None:
        st.stop()
    raw = load_data(up_act, up_sleep)
else:
    raw = load_data(act_path, sleep_path)

# ------------------------------------------------------------------
# 2. 사이드바 설정
# ------------------------------------------------------------------
st.sidebar.header("⚙️ 분석 설정")
threshold = st.sidebar.slider(
    "수면 부족 기준 (분 미만)", 300, 480, 420, 10,
    help="420분 = 7시간. 이 값보다 적게 자면 '수면 부족'으로 분류합니다."
)
st.sidebar.caption(f"기준: {threshold // 60}시간 {threshold % 60}분 미만")

remove_noise = st.sidebar.checkbox(
    "기기를 거의 안 찬 날 제외 (걸음 수 < 500)", value=True
)

df = raw.copy()
if remove_noise:
    df = df[df["TotalSteps"] >= 500]

df["수면부족"] = (df["TotalMinutesAsleep"] < threshold).astype(int)
df["그룹"] = df["수면부족"].map({1: "수면 부족", 0: "충분한 수면"})

st.sidebar.markdown("---")
st.sidebar.metric("분석에 사용된 (활동일→다음날 수면) 쌍", len(df))
st.sidebar.metric("참가자 수", df["Id"].nunique())
st.sidebar.metric("수면 부족 비율", f"{df['수면부족'].mean() * 100:.1f}%")

# ------------------------------------------------------------------
# 3. 메인 화면
# ------------------------------------------------------------------
st.title("😴 하루의 활동 패턴만으로 다음 날 수면 부족을 예측할 수 있을까?")
st.write(
    "Fitbit 데이터를 이용해 **오늘의 활동량**과 **다음 날 기록된 수면 시간**의 관계를 "
    "여러 그래프로 살펴봅니다. 왼쪽에서 기준을 바꿔 보며, 그래프를 골라 결론을 스스로 내려 보세요."
)

chart_options = [
    "1. 수면 시간 분포 (히스토그램)",
    "2. 활동 vs 다음 날 수면 (산점도 + 추세선)",
    "3. 수면 부족 그룹별 활동 비교 (박스플롯)",
    "4. 걸음 수 구간별 수면 부족 비율 (막대그래프)",
    "5. 변수 간 상관관계 (히트맵)",
    "6. 참가자별 비교 (평균 활동 vs 평균 수면)",
    "7. 개인별 시계열 (활동과 수면 함께 보기)",
    "8. 예측 모델 성능 (ROC 곡선 · 혼동행렬)",
]
choice = st.selectbox("📊 보고 싶은 그래프를 선택하세요", chart_options)
st.markdown("---")

# ---------------- 1. 히스토그램 ----------------
if choice == chart_options[0]:
    fig = px.histogram(
        df, x="TotalMinutesAsleep", nbins=30,
        color="그룹",
        color_discrete_map={"수면 부족": "#EF553B", "충분한 수면": "#636EFA"},
        labels={"TotalMinutesAsleep": "수면 시간(분)"},
        title="다음 날 수면 시간 분포",
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                  annotation_text=f"기준 {threshold}분")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💭 생각해 보기: 수면 시간이 아주 짧은(예: 100분 이하) 기록은 진짜 수면 부족일까요, "
            "아니면 낮잠이나 기기 측정 오류일까요?")

# ---------------- 2. 산점도 ----------------
elif choice == chart_options[1]:
    feat = st.selectbox("활동 변수 선택", list(FEATURES.keys()),
                        format_func=lambda x: FEATURES[x])
    color_by = st.radio("점 색상 기준", ["수면 부족 여부", "참가자"], horizontal=True)

    fig = px.scatter(
        df, x=feat, y="TotalMinutesAsleep",
        color="그룹" if color_by == "수면 부족 여부" else "Id",
        color_discrete_map={"수면 부족": "#EF553B", "충분한 수면": "#636EFA"},
        labels={feat: FEATURES[feat], "TotalMinutesAsleep": "다음 날 수면 시간(분)"},
        hover_data=["Id", "ActivityDate"],
        title=f"{FEATURES[feat]} vs 다음 날 수면 시간",
    )
    # 추세선 (numpy)
    x, y = df[feat].values, df["TotalMinutesAsleep"].values
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                             name="추세선", line=dict(color="black", dash="dash")))
    fig.add_hline(y=threshold, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    r = np.corrcoef(x, y)[0, 1]
    st.metric("상관계수 (r)", f"{r:.3f}")
    st.info("💭 생각해 보기: r이 0에 가까우면 어떤 의미일까요? 점들이 넓게 퍼져 있다면 "
            "이 변수 하나로 수면 시간을 맞출 수 있을까요?")

# ---------------- 3. 박스플롯 ----------------
elif choice == chart_options[2]:
    feat = st.selectbox("활동 변수 선택", list(FEATURES.keys()),
                        format_func=lambda x: FEATURES[x])
    fig = px.box(
        df, x="그룹", y=feat, color="그룹", points="all",
        color_discrete_map={"수면 부족": "#EF553B", "충분한 수면": "#636EFA"},
        labels={feat: FEATURES[feat]},
        title=f"수면 부족 여부에 따른 전날 {FEATURES[feat]}",
    )
    st.plotly_chart(fig, use_container_width=True)

    summary = df.groupby("그룹")[feat].agg(["count", "mean", "median"]).round(1)
    summary.columns = ["표본 수", "평균", "중앙값"]
    st.dataframe(summary)
    st.info("💭 생각해 보기: 두 그룹의 상자가 많이 겹친다면, 이 변수로 두 그룹을 구분할 수 있을까요?")

# ---------------- 4. 구간별 막대 ----------------
elif choice == chart_options[3]:
    feat = st.selectbox("구간으로 나눌 변수", list(FEATURES.keys()),
                        format_func=lambda x: FEATURES[x])
    n_bins = st.slider("구간 수", 3, 8, 4)

    tmp = df.copy()
    tmp["구간"] = pd.qcut(tmp[feat], q=n_bins, duplicates="drop")
    rate = tmp.groupby("구간", observed=True).agg(
        수면부족비율=("수면부족", "mean"), 표본수=("수면부족", "size")
    ).reset_index()
    rate["수면부족비율"] = rate["수면부족비율"] * 100
    rate["구간"] = rate["구간"].astype(str)

    fig = px.bar(
        rate, x="구간", y="수면부족비율", text="표본수",
        labels={"수면부족비율": "수면 부족 비율(%)", "구간": FEATURES[feat]},
        title=f"{FEATURES[feat]} 구간별 다음 날 수면 부족 비율 (막대 위 숫자 = 표본 수)",
    )
    fig.add_hline(y=df["수면부족"].mean() * 100, line_dash="dash",
                  annotation_text="전체 평균")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💭 생각해 보기: 구간이 올라갈수록 비율이 일정하게 변하나요? 표본 수가 적은 구간의 "
            "비율은 얼마나 믿을 수 있을까요?")

# ---------------- 5. 히트맵 ----------------
elif choice == chart_options[4]:
    cols = list(FEATURES.keys()) + ["TotalMinutesAsleep"]
    corr = df[cols].corr().round(2)
    names = {**FEATURES, "TotalMinutesAsleep": "다음 날 수면 시간"}
    corr = corr.rename(index=names, columns=names)

    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title="활동 변수와 다음 날 수면 시간의 상관관계",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("💭 생각해 보기: 마지막 행(수면 시간)의 값들이 모두 작다면 어떤 결론을 내릴 수 있을까요? "
            "활동 변수끼리의 상관이 높은 것은 분석에 어떤 문제를 줄까요?")

# ---------------- 6. 참가자별 비교 ----------------
elif choice == chart_options[5]:
    feat = st.selectbox("활동 변수 선택", list(FEATURES.keys()),
                        format_func=lambda x: FEATURES[x])
    per = df.groupby("Id").agg(
        평균활동=(feat, "mean"),
        평균수면=("TotalMinutesAsleep", "mean"),
        일수=("Id", "size"),
    ).reset_index()

    fig = px.scatter(
        per, x="평균활동", y="평균수면", size="일수", hover_name="Id",
        labels={"평균활동": f"평균 {FEATURES[feat]}", "평균수면": "평균 수면 시간(분)"},
        title="참가자별 평균 활동량 vs 평균 수면 시간 (점 크기 = 기록 일수)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("💭 생각해 보기: '사람들 사이의 차이'와 '같은 사람의 날마다의 차이'는 다른 질문입니다. "
            "우리가 알고 싶은 것은 어느 쪽일까요?")

# ---------------- 7. 개인별 시계열 ----------------
elif choice == chart_options[6]:
    counts = df["Id"].value_counts()
    ids = counts.index.tolist()
    pid = st.selectbox("참가자 선택 (기록 많은 순)", ids,
                       format_func=lambda x: f"{x} ({counts[x]}일)")
    feat = st.selectbox("활동 변수 선택", list(FEATURES.keys()),
                        format_func=lambda x: FEATURES[x])
    p = df[df["Id"] == pid].sort_values("ActivityDate")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=p["ActivityDate"], y=p[feat],
                         name=FEATURES[feat], opacity=0.5, yaxis="y1"))
    fig.add_trace(go.Scatter(x=p["ActivityDate"], y=p["TotalMinutesAsleep"],
                             name="다음 날 수면(분)", mode="lines+markers",
                             line=dict(color="crimson"), yaxis="y2"))
    fig.add_hline(y=threshold, line_dash="dot", line_color="gray", yref="y2")
    fig.update_layout(
        title=f"참가자 {pid}: 활동일 기준 활동량(막대)과 그 다음 날 수면(선)",
        yaxis=dict(title=FEATURES[feat]),
        yaxis2=dict(title="수면 시간(분)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("💭 생각해 보기: 활동량이 많은 날 다음에 수면이 항상 늘어나거나 줄어드나요? "
            "참가자마다 패턴이 같은가요?")

# ---------------- 8. 예측 모델 ----------------
elif choice == chart_options[7]:
    st.subheader("로지스틱 회귀로 '내일 수면 부족'을 예측해 보기")
    sel = st.multiselect(
        "모델에 넣을 활동 변수", list(FEATURES.keys()),
        default=["TotalSteps", "VeryActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes"],
        format_func=lambda x: FEATURES[x],
    )

    if len(sel) == 0:
        st.warning("변수를 1개 이상 선택하세요.")
    elif df["수면부족"].nunique() < 2 or df["Id"].nunique() < 5:
        st.warning("데이터가 부족합니다. 기준을 조정해 보세요.")
    else:
        X, y, groups = df[sel], df["수면부족"], df["Id"]

        # 같은 사람의 데이터가 학습/검증에 섞이지 않도록 사람 단위로 분할
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        cv = GroupKFold(n_splits=5)
        proba = cross_val_predict(model, X, y, cv=cv, groups=groups,
                                  method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y, proba)
        acc = accuracy_score(y, pred)
        baseline = max(y.mean(), 1 - y.mean())

        c1, c2, c3 = st.columns(3)
        c1.metric("AUC", f"{auc:.3f}", help="0.5 = 동전 던지기 수준, 1.0 = 완벽")
        c2.metric("정확도", f"{acc * 100:.1f}%")
        c3.metric("기준선 정확도", f"{baseline * 100:.1f}%",
                  help="무조건 '다수 그룹'이라고 찍었을 때의 정확도")

        left, right = st.columns(2)
        with left:
            fpr, tpr, _ = roc_curve(y, proba)
            roc = go.Figure()
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"모델 (AUC={auc:.2f})"))
            roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="무작위", line=dict(dash="dash")))
            roc.update_layout(title="ROC 곡선",
                              xaxis_title="거짓 양성 비율", yaxis_title="참 양성 비율")
            st.plotly_chart(roc, use_container_width=True)

        with right:
            cm = confusion_matrix(y, pred)
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=["예측: 충분", "예측: 부족"], y=["실제: 충분", "실제: 부족"],
                title="혼동행렬",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # 계수 (어떤 변수가 영향이 큰가)
        model.fit(X, y)
        coef = pd.DataFrame({
            "변수": [FEATURES[s] for s in sel],
            "계수": model.named_steps["logisticregression"].coef_[0],
        }).sort_values("계수")
        fig_coef = px.bar(coef, x="계수", y="변수", orientation="h",
                          title="변수별 계수 (양수 = 수면 부족 가능성 ↑)")
        st.plotly_chart(fig_coef, use_container_width=True)

        st.info("💭 생각해 보기: AUC가 0.5에 가깝다면 활동 패턴만으로 예측하기 어렵다는 뜻일까요? "
                "정확도와 기준선 정확도를 비교하면 모델이 정말 쓸모 있는지 알 수 있을까요?")

# ------------------------------------------------------------------
# 4. 한계점
# ------------------------------------------------------------------
st.markdown("---")
with st.expander("⚠️ 이 분석의 한계 (결론 내리기 전에 꼭 읽어보세요)"):
    st.markdown(
        """
- **표본이 작습니다.** 참가자 수가 적고, 수면 기록이 없는 날이 많아 분석에 쓰인 날 수가 제한적입니다.
- **수면 기록의 정의.** 낮잠이나 짧은 기록도 하루 수면으로 잡힐 수 있습니다.
- **상관관계 ≠ 인과관계.** 활동이 수면에 영향을 준 것인지, 다른 요인(시험, 스트레스, 생활 습관)이 있는지는 알 수 없습니다.
- **기기 착용 여부.** 기기를 안 찬 날은 활동이 낮게 기록됩니다. (사이드바에서 제외 가능)
- **참가자 차이.** 사람마다 기본 수면 패턴이 다르므로 개인차가 결과를 흐릴 수 있습니다.
        """
    )
