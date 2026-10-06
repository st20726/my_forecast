import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균기온 선형회귀 모델")
st.write(
    "과거 서울의 연평균기온을 이용하여 선형회귀 모델을 만들고 "
    "최근 20년의 기온을 얼마나 잘 예측하는지 평가합니다."
)


# =========================================================
# 데이터 주소
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 결측값 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 연도
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()

except Exception as e:

    st.error("기온 데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 2025년까지만 사용
# =========================================================

df = df[
    df["연도"] <= 2025
].copy()


# =========================================================
# 연도별 연평균기온 계산
# =========================================================

yearly = (
    df.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)


# =========================================================
# 관측일 300일 미만인 연도 제외
# =========================================================

yearly = yearly[
    yearly["관측일수"] >= 300
].copy()

yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# =========================================================
# 데이터가 충분한지 확인
# =========================================================

if len(yearly) < 2:

    st.error(
        "연평균기온을 계산할 수 있는 데이터가 충분하지 않습니다."
    )

    st.stop()


# =========================================================
# 독립변수
#
# 연도 자체를 사용하면 큰 숫자가 되므로
# 1908년을 기준으로 지난 연수를 사용
#
# 1908년 → 0
# 1909년 → 1
# ...
# =========================================================

yearly["지난연수"] = yearly["연도"] - 1908


# =========================================================
# 학습 / 테스트 기간
# =========================================================

TEST_START = 2006
TEST_END = 2025

RECENT_50_START = 1956
RECENT_50_END = 2005

RECENT_100_START = 1906
RECENT_100_END = 2005


# =========================================================
# 데이터 분리
# =========================================================

train_50 = yearly[
    (yearly["연도"] >= RECENT_50_START)
    & (yearly["연도"] <= RECENT_50_END)
].copy()


train_100 = yearly[
    (yearly["연도"] >= RECENT_100_START)
    & (yearly["연도"] <= RECENT_100_END)
].copy()


test = yearly[
    (yearly["연도"] >= TEST_START)
    & (yearly["연도"] <= TEST_END)
].copy()


# =========================================================
# 데이터 존재 여부 확인
# =========================================================

if len(train_50) < 2:

    st.error(
        "1956~2005년 훈련 데이터가 충분하지 않습니다."
    )

    st.stop()


if len(train_100) < 2:

    st.error(
        "1906~2005년 훈련 데이터가 충분하지 않습니다."
    )

    st.stop()


if len(test) < 2:

    st.error(
        "2006~2025년 테스트 데이터가 충분하지 않습니다."
    )

    st.stop()


# =========================================================
# 선형회귀 함수
# =========================================================

def make_model(train_data):

    X = train_data[
        ["지난연수"]
    ]

    y = train_data[
        "연평균기온"
    ]

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    return model


# =========================================================
# 모델 만들기
# =========================================================

model_50 = make_model(
    train_50
)

model_100 = make_model(
    train_100
)


# =========================================================
# 테스트 데이터 예측
# =========================================================

X_test = test[
    ["지난연수"]
]


y_test = test[
    "연평균기온"
].to_numpy()


prediction_50 = model_50.predict(
    X_test
)

prediction_100 = model_100.predict(
    X_test
)


# =========================================================
# 평가 지표 계산 함수
# =========================================================

def evaluate_model(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, mse, r2


# =========================================================
# 50년 모델 평가
# =========================================================

mae_50, mse_50, r2_50 = evaluate_model(
    y_test,
    prediction_50
)


# =========================================================
# 100년 모델 평가
# =========================================================

mae_100, mse_100, r2_100 = evaluate_model(
    y_test,
    prediction_100
)


# =========================================================
# 기울기 계산
#
# sklearn LinearRegression의 coef_는
# 1년당 기온 변화량
#
# ×100 → 100년당 기온 변화량
# =========================================================

slope_50_year = model_50.coef_[0]
slope_100_year = model_100.coef_[0]

slope_50_100 = slope_50_year * 100
slope_100_100 = slope_100_year * 100


# =========================================================
# 모델 정보
# =========================================================

st.subheader("📚 학습 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("### 최근 50년 학습")

    st.write(
        f"**{RECENT_50_START}~{RECENT_50_END}년**"
    )

    st.write(
        f"사용 연도: **{len(train_50)}년**"
    )


with col2:

    st.markdown("### 최근 100년 학습")

    actual_100_start = int(
        train_100["연도"].min()
    )

    st.write(
        f"요청 기간: **{RECENT_100_START}~{RECENT_100_END}년**"
    )

    st.write(
        f"실제 사용 시작: **{actual_100_start}년**"
    )

    st.write(
        f"사용 연도: **{len(train_100)}년**"
    )


with col3:

    st.markdown("### 공통 테스트")

    st.write(
        f"**{TEST_START}~{TEST_END}년**"
    )

    st.write(
        f"사용 연도: **{len(test)}년**"
    )


st.info(
    "두 모델 모두 2006~2025년을 한 번도 학습하지 않고 "
    "테스트 데이터로만 사용합니다."
)


# =========================================================
# 기울기 비교
# =========================================================

st.subheader("🌡️ 회귀선의 기울기 비교")

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "최근 50년 모델",
        f"{slope_50_100:+.2f} ℃ / 100년"
    )

    st.caption(
        "1956~2005년 데이터로 학습"
    )


with col2:

    st.metric(
        "최근 100년 모델",
        f"{slope_100_100:+.2f} ℃ / 100년"
    )

    st.caption(
        "1906~2005년 데이터로 학습"
    )


# =========================================================
# 기울기 차이
# =========================================================

slope_difference = (
    slope_50_100
    - slope_100_100
)


st.write(
    f"두 모델의 기울기 차이는 "
    f"**{slope_difference:+.2f} ℃ / 100년**입니다."
)


# =========================================================
# 테스트 성능 비교
# =========================================================

st.subheader("🎯 2006~2025년 예측 성능 비교")

comparison = pd.DataFrame({

    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],

    "학습기간": [
        "1956~2005",
        "1906~2005"
    ],

    "MAE (℃)": [
        mae_50,
        mae_100
    ],

    "MSE (℃²)": [
        mse_50,
        mse_100
    ],

    "R²": [
        r2_50,
        r2_100
    ],

    "기울기 (℃/100년)": [
        slope_50_100,
        slope_100_100
    ]
})


st.dataframe(
    comparison.style.format({

        "MAE (℃)": "{:.3f}",

        "MSE (℃²)": "{:.3f}",

        "R²": "{:.3f}",

        "기울기 (℃/100년)": "{:+.2f}"

    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 어떤 모델이 더 좋은지 자동 판단
# =========================================================

st.subheader("🏆 테스트 성능 비교 결과")


if mae_50 < mae_100:

    st.write(
        f"**MAE:** 최근 50년 모델이 더 작습니다. "
        f"({mae_50:.3f}℃ < {mae_100:.3f}℃)"
    )

else:

    st.write(
        f"**MAE:** 최근 100년 모델이 더 작습니다. "
        f"({mae_100:.3f}℃ < {mae_50:.3f}℃)"
    )


if mse_50 < mse_100:

    st.write(
        f"**MSE:** 최근 50년 모델이 더 작습니다. "
        f"({mse_50:.3f} < {mse_100:.3f})"
    )

else:

    st.write(
        f"**MSE:** 최근 100년 모델이 더 작습니다. "
        f"({mse_100:.3f} < {mse_50:.3f})"
    )


if r2_50 > r2_100:

    st.write(
        f"**R²:** 최근 50년 모델이 더 높습니다. "
        f"({r2_50:.3f} > {r2_100:.3f})"
    )

else:

    st.write(
        f"**R²:** 최근 100년 모델이 더 높습니다. "
        f"({r2_100:.3f} > {r2_50:.3f})"
    )


# =========================================================
# 테스트 데이터 실제값 vs 예측값
# =========================================================

st.subheader(
    "📈 테스트 데이터에서 실제 기온과 예측 기온 비교"
)

fig_test = go.Figure()


# 실제 기온
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        line=dict(
            width=3
        ),
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=prediction_50,
        mode="lines",
        name="최근 50년 학습 모델",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "50년 모델 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 100년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=prediction_100,
        mode="lines",
        name="최근 100년 학습 모델",
        line=dict(
            width=3,
            dash="dot"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "100년 모델 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig_test.update_layout(

    xaxis=dict(
        title="연도",
        tickmode="linear",
        dtick=1
    ),

    yaxis=dict(
        title="연평균기온 (℃)"
    ),

    height=600,

    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)


st.plotly_chart(
    fig_test,
    use_container_width=True
)


# =========================================================
# 회귀선 전체 모습
# =========================================================

st.subheader("📊 학습 데이터와 회귀선")

fig_models = go.Figure()


# 전체 연평균기온
fig_models.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=5
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 회귀선
model_years_50 = np.linspace(
    int(train_50["연도"].min()),
    TEST_END,
    300
)

model_x_50 = (
    model_years_50 - 1908
)

model_y_50 = model_50.predict(
    model_x_50.reshape(-1, 1)
)


fig_models.add_trace(
    go.Scatter(
        x=model_years_50,
        y=model_y_50,
        mode="lines",
        name="최근 50년 학습 회귀선",
        line=dict(
            width=4,
            dash="dash"
        )
    )
)


# 100년 회귀선
model_years_100 = np.linspace(
    int(train_100["연도"].min()),
    TEST_END,
    300
)

model_x_100 = (
    model_years_100 - 1908
)

model_y_100 = model_100.predict(
    model_x_100.reshape(-1, 1)
)


fig_models.add_trace(
    go.Scatter(
        x=model_years_100,
        y=model_y_100,
        mode="lines",
        name="최근 100년 학습 회귀선",
        line=dict(
            width=4,
            dash="dot"
        )
    )
)


# 테스트 구간 시작선
fig_models.add_vline(
    x=2006,
    line_width=2,
    line_dash="dash"
)


fig_models.add_annotation(
    x=2006,
    y=1,
    yref="paper",
    text="테스트 시작",
    showarrow=False,
    yanchor="bottom"
)


fig_models.update_layout(

    xaxis=dict(
        title="연도",
        type="linear",
        dtick=10
    ),

    yaxis=dict(
        title="연평균기온 (℃)"
    ),

    height=650,

    hovermode="closest",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)


st.plotly_chart(
    fig_models,
    use_container_width=True
)


# =========================================================
# 평가 지표 설명
# =========================================================

st.subheader("📖 평가 지표 읽는 법")

st.write(
    "**MAE (평균 절대 오차)**: 실제 기온과 예측 기온이 평균적으로 "
    "몇 ℃ 정도 차이 나는지를 나타냅니다. **작을수록 좋습니다.**"
)

st.write(
    "**MSE (평균 제곱 오차)**: 예측 오차를 제곱해서 평균낸 값입니다. "
    "큰 오차에 더 큰 벌점을 줍니다. **작을수록 좋습니다.**"
)

st.write(
    "**R² (결정계수)**: 모델이 실제 기온의 변동을 얼마나 설명하는지를 "
    "나타냅니다. 일반적으로 **1에 가까울수록 좋습니다.**"
)


# =========================================================
# 데이터 처리 기준
# =========================================================

st.subheader("📌 데이터 처리 기준")

st.write(
    "• 2025년까지의 데이터만 사용했습니다."
)

st.write(
    "• 연간 관측일이 300일 미만인 연도는 제외했습니다."
)

st.write(
    "• 일평균기온을 연도별로 평균내어 연평균기온을 만들었습니다."
)

st.write(
    "• 최근 50년 모델은 1956~2005년으로 학습했습니다."
)

st.write(
    "• 최근 100년 모델은 1906~2005년으로 학습했습니다."
)

st.write(
    "• 두 모델 모두 2006~2025년을 공통 테스트 데이터로 사용했습니다."
)

st.write(
    "• 회귀의 독립변수는 `연도 - 1908`이며, "
    "기울기는 이해하기 쉽도록 100년 단위로 환산했습니다."
)
