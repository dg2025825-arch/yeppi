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

# ------------------------------------------------------------------
# 1. 데이터 불러오기
# ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent      # .../code
REPO_ROOT = HERE.parent                     # 저장소 최상위


def find_file(keyword):
    """data 폴더 → 저장소 전체 → 현재 작업 폴더 순서로 CSV를 찾는다."""
    search_dirs = [REPO_ROOT / "data", REPO_ROOT, HERE, Path.cwd()]
    for d in search_dirs:
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


def get_data():
    act_path = find_file("dailyactivity")
    sleep_path = find_file("sleepday")
    if act_path is not None and sleep_path is not None:
        return load_data(str(act_path), str(sleep_path)), f"{act_path.parent.name}/ 폴더 ({act_path.name}, {sleep_path.name})"

    st.error("저장소에서 CSV 파일을 찾지 못했습니다. 아래 정보를 확인하거나 직접 올려 주세요.")
    with st.expander("🔎 앱이 보고 있는 폴더와 파일 목록", expanded=True):
        st.write("main.py 위치:", str(HERE))
        st.write("저장소 최상위로 추정한 위치:", str(REPO_ROOT))
        files = sorted(
            str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts
        )
        st.write(files if files else "파일이 없습니다.")
    up_act = st.file_uploader(f"{ACT_NAME} 업로드", type="csv")
    up_sleep = st.file_uploader(f"{SLEEP_NAME} 업로드", type="csv")
    if up_act is None or up_sleep is None:
        st.stop()
    return load_data(up_act, up_sleep), "직접 업로드"


raw, source = get_data()

# ------------------------------------------------------------------
# 2. 그래프 해설 텍스트
# ------------------------------------------------------------------
GUIDE = {
    0: {
        "fit": "⭐⭐ (보조) — 결과 변수인 수면의 모양을 이해하는 그래프",
        "what": """
**무엇을 보여주나요?** 다음 날 수면 시간이 어떤 값에 몰려 있는지, 우리가 정한 '수면 부족 기준선'이 분포의 어디에 걸리는지 보여줍니다.

**읽는 법**
- 막대가 높은 곳 = 그 수면 시간이 자주 나타난다는 뜻입니다.
- 검은 점선(기준선)의 왼쪽이 '수면 부족', 오른쪽이 '충분한 수면'입니다.
        """,
        "know": """
- 대부분의 날이 몇 시간 수면 근처에 모여 있는지
- 기준선을 옮기면 수면 부족 비율이 얼마나 민감하게 바뀌는지
- 매우 짧은 수면(예: 100분 이하) 같은 이상치가 있는지
        """,
        "limit": """
- **활동 변수가 전혀 들어 있지 않아서** 질문에 대한 답은 직접 주지 못합니다.
- 기준선(7시간 등)은 연구자가 정한 값이어서, 기준에 따라 '수면 부족' 비율이 크게 달라질 수 있습니다.
- 아주 짧은 기록은 진짜 수면 부족이 아니라 **낮잠이나 기기 측정 오류**일 수 있습니다.
        """,
        "use": "분석을 시작하기 전에 '수면 부족'이라는 라벨이 얼마나 믿을 만한지 점검하는 용도로 쓰세요.",
    },
    1: {
        "fit": "⭐⭐⭐⭐ — 활동량과 다음 날 수면 시간의 관계 강도를 직접 확인",
        "what": """
**무엇을 보여주나요?** 한 점 = (참가자, 하루). x축은 그날의 활동 변수, y축은 **그 다음 날** 수면 시간입니다. 점선은 추세선(직선 관계), 아래 숫자 r은 상관계수입니다.

**읽는 법**
- 점들이 우상향/우하향 띠 모양이면 관계가 있다는 뜻이고, 구름처럼 퍼져 있으면 관계가 약하다는 뜻입니다.
- r은 -1~1 사이 값입니다. 0에 가까우면 직선 관계가 약합니다.
        """,
        "know": """
- 활동이 많을수록 다음 날 수면이 늘거나 줄어드는 **경향**이 있는지
- 색상을 '참가자'로 바꾸면, 같은 사람의 점들이 한 덩어리로 모이는지(개인차가 큰지)
- 이상치(예: 하루에 걸음 수가 비정상적으로 많은 날)가 추세선을 끌고 가는지
        """,
        "limit": """
- r이 작다고 해서 '아무 관계도 없다'는 뜻은 아닙니다. **직선이 아닌 관계**(너무 적거나 너무 많을 때만 영향)는 r에 잡히지 않습니다.
- 이상치 몇 개가 추세선과 r을 크게 바꿀 수 있습니다.
- 같은 사람의 여러 날이 섞여 있어서 점들이 **서로 독립이 아닙니다.** 그래서 우연히 보이는 관계에 과신하기 쉽습니다.
- 수면 시간(연속값)을 보는 그래프라서, '수면 부족인지 아닌지'라는 이진 질문과는 약간 다릅니다.
        """,
        "use": "활동 변수 6개를 하나씩 바꿔 보며 '가장 관계가 뚜렷한 변수'를 찾는 데 적합합니다. 단, 관계가 뚜렷해도 예측이 잘 된다는 뜻은 아닙니다. 8번에서 확인하세요.",
    },
    2: {
        "fit": "⭐⭐⭐⭐ — '수면 부족이었던 날'과 '충분했던 날'의 전날 활동 차이 비교",
        "what": """
**무엇을 보여주나요?** 다음 날 수면 부족 여부로 두 그룹을 나눈 뒤, 두 그룹의 **전날 활동량 분포**를 나란히 비교합니다. 상자 안의 선이 중앙값, 상자는 중간 50% 범위입니다.

**읽는 법**
- 두 상자가 **많이 겹치면** 이 변수로 두 그룹을 구분하기 어렵다는 뜻입니다.
- 두 상자가 위아래로 **뚜렷이 갈라지면** 예측에 쓸모 있는 변수일 가능성이 있습니다.
        """,
        "know": """
- 예측 질문과 같은 방향(활동 → 수면 부족 여부)으로 차이를 비교할 수 있습니다.
- 평균 말고 중앙값, 퍼짐 정도, 이상치를 한꺼번에 볼 수 있습니다.
- 아래 표의 **표본 수**로 어느 그룹이 얼마나 적은지 확인할 수 있습니다.
        """,
        "limit": """
- 그룹 간 **평균 차이가 있어도** 겹침이 크면 개별 날짜를 예측하기에는 부족합니다.
- 수면 부족 그룹의 표본 수가 적으면 비교가 불안정합니다.
- 변수를 **하나씩만** 비교하므로 여러 변수를 조합했을 때의 효과는 볼 수 없습니다.
- 사람마다 기본 수면량이 달라서, 특정 몇 명이 한 그룹을 대부분 차지하면 결과가 왜곡될 수 있습니다.
        """,
        "use": "'차이가 있다'와 '구분할 수 있다'는 다른 말입니다. 겹침의 정도를 꼭 확인하세요.",
    },
    3: {
        "fit": "⭐⭐⭐ — 활동량이 늘어날 때 수면 부족 비율이 어떻게 변하는지 확인",
        "what": """
**무엇을 보여주나요?** 활동 변수를 크기 순으로 같은 인원씩 나눈 구간별로, **다음 날 수면 부족이었던 비율(%)**을 막대로 보여줍니다. 막대 위 숫자는 표본 수이고, 점선은 전체 평균입니다.

**읽는 법**
- 막대가 구간이 올라갈수록 **일정하게 올라가거나 내려가면** 방향성이 있다는 뜻입니다.
- 들쭉날쭉하면 우연의 가능성이 큽니다.
        """,
        "know": """
- 직선이 아닌 **비선형 관계**(너무 적거나 너무 많을 때만 수면 부족 증가 등)도 볼 수 있습니다.
- 비율이라서 이진 질문(수면 부족 여부)에 맞는 표현입니다.
        """,
        "limit": """
- **구간 수에 따라 모양이 바뀝니다.** 구간 수 슬라이더를 바꿔도 같은 경향이 유지되는지 확인해야 합니다.
- 표본 수가 적은 구간의 비율은 한두 건에도 크게 흔들립니다. (막대 위 숫자 확인!)
- 신뢰구간이 표시되지 않아서 '차이가 우연인지'는 알 수 없습니다.
- 변수를 하나씩만 봅니다.
        """,
        "use": "방향성(단조 증가/감소)이 보이는지, 구간 수를 바꿔도 유지되는지가 핵심입니다.",
    },
    4: {
        "fit": "⭐⭐⭐ — 전체 변수의 관계를 한눈에 훑어보는 지도",
        "what": """
**무엇을 보여주나요?** 활동 변수들과 다음 날 수면 시간 사이의 상관계수를 색으로 나타낸 표입니다. 파랑 = 음의 상관, 빨강 = 양의 상관, 흰색 = 거의 없음.

**읽는 법**
- 가장 아래(또는 오른쪽) '다음 날 수면 시간' 행/열이 핵심입니다. 이 값들이 모두 0에 가까우면 개별 변수 하나하나는 수면 시간과 약한 관계라는 뜻입니다.
        """,
        "know": """
- 어떤 변수가 수면과 상대적으로 더 관련이 큰지 순위를 매길 수 있습니다.
- **활동 변수끼리 서로 높게 상관**되어 있는지 알 수 있습니다. (예: 걸음 수와 활동 시간)
        """,
        "limit": """
- 상관계수는 **직선 관계만** 측정하고 이상치에 민감합니다.
- 활동 변수끼리 상관이 높으면(다중공선성) 모델에서 각 변수의 역할을 따로 해석하기 어렵습니다. '활동 시간 합계'는 고강도+중강도를 더한 변수라서 원래 변수들과 겹칩니다.
- 상관이 있다고 해서 **원인**이라는 뜻은 아닙니다.
- 수면 시간(연속값)의 상관이지, 수면 부족 여부에 대한 직접적인 예측력이 아닙니다.
        """,
        "use": "분석 초반에 후보 변수를 고르고, 변수들이 서로 겹치는지 확인하는 용도로 적합합니다.",
    },
    5: {
        "fit": "⭐ — 질문과는 다른 수준(사람 간 차이)의 분석. 해석에 주의",
        "what": """
**무엇을 보여주나요?** 참가자 한 명을 점 하나로 요약해서 **평균 활동량 vs 평균 수면 시간**을 비교합니다. 점이 클수록 기록 일수가 많습니다.

**읽는 법**
- 활동량이 많은 사람들이 평균적으로 더 오래/적게 자는 경향이 있는지 봅니다.
        """,
        "know": """
- 사람 간 차이(활동적인 사람 vs 비활동적인 사람)의 경향
- 기록 일수가 적은 참가자(작은 점)는 평균이 불안정하다는 점
        """,
        "limit": """
- 우리 질문은 **"같은 사람이 오늘 활동이 많으면 내일 수면이 달라지나?"**(개인 내 변화)인데, 이 그래프는 **"활동이 많은 사람은 수면이 다른가?"**(사람 간 차이)입니다. 두 질문의 답은 서로 다를 수 있습니다. (이를 '생태학적 오류'라고 부르기도 합니다.)
- 참가자 수가 적어서 점이 몇 개 안 되고, 점 하나가 경향을 좌우합니다.
- 평균으로 요약하면 날마다의 변동 정보가 사라집니다.
        """,
        "use": "'사람 간 차이'와 '개인 내 변화'를 구분해서 생각해 보는 **비판적 사고용 그래프**입니다. 이 그래프만으로 결론을 내리면 안 됩니다.",
    },
    6: {
        "fit": "⭐⭐ — 개인 안에서의 패턴을 눈으로 확인하는 탐색용",
        "what": """
**무엇을 보여주나요?** 한 참가자의 날짜별 활동량(막대)과 **그 다음 날** 수면 시간(선)을 같은 시간축에 겹쳐 보여줍니다. 오른쪽 축이 수면 시간입니다.

**읽는 법**
- 막대가 높은 날의 다음 날 선이 어떻게 움직이는지 봅니다.
- 참가자를 바꿔서 패턴이 비슷한지 비교합니다.
        """,
        "know": """
- 개인 안에서의 **날마다 변화**를 볼 수 있어서 우리 질문의 '개인 내 관점'과 가장 가깝습니다.
- 참가자마다 패턴이 일관되는지, 제각각인지 확인할 수 있습니다.
- 기록이 빠진 날(수면 기록 없음)이 얼마나 많은지도 보입니다.
        """,
        "limit": """
- 한 사람의 몇 주 데이터는 **사례 관찰**일 뿐이고, 일반화할 수 없습니다.
- 두 축의 눈금을 임의로 맞춰서 **관계가 있어 보이게(또는 없어 보이게)** 착시가 생길 수 있습니다.
- 정량적인 결론(r, 정확도)은 주지 않습니다.
        """,
        "use": "다른 그래프에서 나온 결과가 개인 단위에서도 보이는지 **예시로 확인**하는 용도입니다. 여러 참가자를 비교해 보세요.",
    },
    7: {
        "fit": "⭐⭐⭐⭐⭐ — 질문('예측할 수 있는가')에 가장 직접적으로 답하는 그래프",
        "what": """
**무엇을 보여주나요?** 활동 변수만으로 '내일 수면 부족 여부'를 예측하는 로지스틱 회귀 모델을 만들고 그 성능을 평가합니다.

- **AUC**: 0.5 = 동전 던지기 수준, 1.0 = 완벽. 수면 부족인 날이 아닌 날보다 높은 점수를 받을 확률입니다.
- **정확도 vs 기준선 정확도**: 아무 정보 없이 다수 그룹으로만 찍어도 나오는 정확도(기준선)보다 높아야 쓸모가 있습니다.
- **ROC 곡선**: 대각선(무작위)에서 얼마나 위로 떨어져 있는지 봅니다.
- **혼동행렬**: 어떤 종류의 실수(놓침/오경보)를 많이 하는지 봅니다.
- **계수 그래프**: 어떤 변수가 예측에 기여했는지 방향을 보여줍니다.
        """,
        "know": """
- 여러 활동 변수를 **함께** 썼을 때의 예측력
- 한 번도 보지 못한 사람에게도 통하는지 (같은 참가자가 학습과 검증에 섞이지 않게 `GroupKFold`로 분할했기 때문)
- 변수를 더하거나 빼면 성능이 어떻게 변하는지
        """,
        "limit": """
- 표본이 작아서 **성능 숫자가 불안정**합니다. 수면 기준이나 제외 옵션을 바꾸면 값이 꽤 달라질 수 있습니다.
- 정확도는 클래스 비율에 속기 쉬워서 **반드시 기준선과 함께**, 그리고 AUC를 우선해서 봐야 합니다.
- 단순한 선형 모델이라 복잡한 관계는 못 잡습니다. 성능이 낮은 것이 '활동에 정보가 없다'는 증명은 아닙니다.
- 이 모델은 **인과관계**가 아니라 예측 성능만 평가합니다.
- 수면 기록이 있는 날만 쓰므로, 기록을 남기지 않은 날이 빠져 **선택 편향**이 있을 수 있습니다.
        """,
        "use": "결론은 AUC(0.5와의 거리), 정확도와 기준선의 차이, 혼동행렬의 실수 유형을 **함께** 보고 내리세요. 2·3번 그래프에서 본 관계와 일관되는지도 확인하세요.",
    },
}


def show_guide(i):
    g = GUIDE[i]
    st.markdown("---")
    st.markdown(f"### 📖 이 그래프 해설  \n**질문 적합도:** {g['fit']}")
    t1, t2, t3, t4 = st.tabs(["📌 무엇을 보여주나", "✅ 알 수 있는 것", "⚠️ 한계", "🎯 활용 팁"])
    with t1:
        st.markdown(g["what"])
    with t2:
        st.markdown(g["know"])
    with t3:
        st.markdown(g["limit"])
    with t4:
        st.markdown(g["use"])


# ------------------------------------------------------------------
# 3. 사이드바 설정
# ------------------------------------------------------------------
st.sidebar.header("⚙️ 분석 설정")
st.sidebar.caption(f"📂 데이터 출처: {source}")

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
# 4. 메인 화면
# ------------------------------------------------------------------
st.title("😴 하루의 활동 패턴만으로 다음 날 수면 부족을 예측할 수 있을까?")
st.write(
    "Fitbit 데이터를 이용해 **오늘의 활동량**과 **다음 날 기록된 수면 시간**의 관계를 "
    "여러 그래프로 살펴봅니다. 왼쪽에서 기준을 바꿔 보며, 그래프를 골라 결론을 스스로 내려 보세요."
)

with st.expander("🧭 어떤 그래프가 우리 질문에 적합할까? (먼저 읽어보세요)", expanded=False):
    st.markdown(
        """
**질문의 핵심 3요소**
1. **활동만으로** → 활동 변수만 사용 (어젯밤 수면, 개인 습관 등은 사용 안 함)
2. **다음 날** → 오늘 활동 → 내일 수면의 시간 순서
3. **예측할 수 있는가** → '관계가 있다'와 '예측이 된다'는 다릅니다

| 그래프 | 적합도 | 역할 |
|---|---|---|
| 8. 예측 모델 | ⭐⭐⭐⭐⭐ | 질문에 가장 직접적으로 답함 |
| 2. 산점도 | ⭐⭐⭐⭐ | 관계의 강도 확인 |
| 3. 박스플롯 | ⭐⭐⭐⭐ | 그룹 간 활동 차이 확인 |
| 4. 구간별 막대 | ⭐⭐⭐ | 방향성·비선형 관계 확인 |
| 5. 히트맵 | ⭐⭐⭐ | 전체 변수 훑어보기 |
| 7. 개인별 시계열 | ⭐⭐ | 개인 내 패턴 사례 관찰 |
| 1. 히스토그램 | ⭐⭐ | 수면 라벨 점검 |
| 6. 참가자별 비교 | ⭐ | 다른 수준의 질문(사람 간 차이) |

**추천 탐구 순서:** 1 → 5 → 2·3 → 4 → 8 → 6·7

**기억할 점:** 하나의 그래프만으로 결론을 내리지 말고, 여러 그래프가 **같은 방향을 가리키는지** 확인하세요.
        """
    )

chart_options = [
    "1. 수면 시간 분포 (히스토그램)",
    "2. 활동 vs 다음 날 수면 (산점도 + 추세선)",
    "3. 수면 부족 그룹별 활동 비교 (박스플롯)",
    "4. 활동량 구간별 수면 부족 비율 (막대그래프)",
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
        df, x="TotalMinutesAsleep", nbins=30, color="그룹",
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
            "비율은 얼마나 믿을 수 있을까요? 구간 수를 바꿔도 같은 경향인가요?")

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

# ---------------- 선택한 그래프의 해설 표시 ----------------
show_guide(chart_options.index(choice))

# ------------------------------------------------------------------
# 5. 전체 한계점
# ------------------------------------------------------------------
st.markdown("---")
with st.expander("⚠️ 이 분석 전체의 한계 (결론 내리기 전에 꼭 읽어보세요)"):
    st.markdown(
        """
- **표본이 작습니다.** 참가자 수가 적고, 수면 기록이 없는 날이 많아 분석에 쓰인 날 수가 제한적입니다.
- **선택 편향.** 수면 기록이 있는 날만 분석에 들어갑니다. 기록을 남기는 날과 안 남기는 날이 다를 수 있습니다.
- **수면 기록의 정의.** 낮잠이나 짧은 기록도 하루 수면으로 잡힐 수 있습니다.
- **기준선은 임의적입니다.** 7시간이라는 기준은 바꿀 수 있고, 기준에 따라 결과가 달라집니다.
- **상관관계 ≠ 인과관계.** 활동이 수면에 영향을 준 것인지, 다른 요인(시험, 스트레스, 생활 습관)이 있는지는 알 수 없습니다.
- **기기 착용 여부.** 기기를 안 찬 날은 활동이 낮게 기록됩니다. (사이드바에서 제외 가능)
- **참가자 차이.** 사람마다 기본 수면 패턴이 다르므로 개인차가 결과를 흐릴 수 있습니다.
- **변수 간 중복.** 활동 변수들이 
