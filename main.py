import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균기온 곡선 예측기")

st.write(
    "과거 서울의 연평균기온을 이용해 1차, 3차, 9차 다항회귀 모델을 만들고 "
    "학습에 사용하지 않은 최근 데이터를 이용해 성능을 비교합니다."
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

    # 평균기온 숫자 변환
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

    st.error(
        "기온 데이터를 불러오는 중 오류가 발생했습니다."
    )

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
# 학습 / 테스트 분리
#
# 2005년 이전 → 훈련
# 2005년부터 → 테스트
# =========================================================

TRAIN_END = 2004
TEST_START = 2005
TEST_END = 2025

train = yearly[
    yearly["연도"] <= TRAIN_END
].copy()

test = yearly[
    (yearly["연도"] >= TEST_START)
    & (yearly["연도"] <= TEST_END)
].copy()


# =========================================================
# 데이터 개수 확인
# =========================================================

if len(train) < 10:

    st.error(
        "훈련 데이터가 충분하지 않습니다."
    )

    st.stop()


if len(test) < 2:

    st.error(
        "테스트 데이터가 충분하지 않습니다."
    )

    st.stop()


# =========================================================
# 고차 다항식의 수치 불안정 방지
#
# 실제 연도 대신
#
# 변환연도 = 연도 - 2005
#
# 를 사용한다.
#
# 예:
# 1905 → -100
# 2000 → -5
# 2005 → 0
# 2025 → 20
# 2050 → 45
#
# 따라서 9차식에서도 숫자가 지나치게 커지지 않는다.
# =========================================================

train["변환연도"] = (
    train["연도"] - TEST_START
)

test["변환연도"] = (
    test["연도"] - TEST_START
)


# =========================================================
# X / y
# =========================================================

X_train = train[
    ["변환연도"]
]

y_train = train[
    "연평균기온"
]

X_test = test[
    ["변환연도"]
]

y_test = test[
    "연평균기온"
]


# =========================================================
# 다항회귀 모델 생성
# =========================================================

def make_polynomial_model(degree):

    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# =========================================================
# 1차 / 3차 / 9차 모델 학습
#
# 중요:
# 모든 모델은 TRAIN 데이터만 사용한다.
# =========================================================

model_1 = make_polynomial_model(1)

model_3 = make_polynomial_model(3)

model_9 = make_polynomial_model(9)


# =========================================================
# 테스트 데이터 예측
# =========================================================

pred_1 = model_1.predict(
    X_test
)

pred_3 = model_3.predict(
    X_test
)

pred_9 = model_9.predict(
    X_test
)


# =========================================================
# MAE 계산
#
# 테스트 데이터에 대해서만 평가
# =========================================================

mae_1 = mean_absolute_error(
    y_test,
    pred_1
)

mae_3 = mean_absolute_error(
    y_test,
    pred_3
)

mae_9 = mean_absolute_error(
    y_test,
    pred_9
)


# =========================================================
# 2050년 예측
# =========================================================

year_2050 = pd.DataFrame({
    "변환연도": [2050 - TEST_START]
})


prediction_2050_1 = model_1.predict(
    year_2050
)[0]

prediction_2050_3 = model_3.predict(
    year_2050
)[0]

prediction_2050_9 = model_9.predict(
    year_2050
)[0]


# =========================================================
# 모델별 결과 정리
# =========================================================

results = pd.DataFrame({

    "모델": [
        "1차 (직선)",
        "3차 곡선",
        "9차 곡선"
    ],

    "차수": [
        1,
        3,
        9
    ],

    "테스트 평균 오차 MAE (℃)": [
        mae_1,
        mae_3,
        mae_9
    ],

    "2050년 예측기온 (℃)": [
        prediction_2050_1,
        prediction_2050_3,
        prediction_2050_9
    ]
})


# =========================================================
# 학습 / 테스트 데이터 개수
# =========================================================

st.subheader("📚 학습 데이터와 테스트 데이터")

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "훈련용 연도 수",
        f"{len(train)}개"
    )

    st.write(
        f"훈련 기간: "
        f"**{int(train['연도'].min())}~"
        f"{int(train['연도'].max())}년**"
    )


with col2:

    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개"
    )

    st.write(
        f"테스트 기간: "
        f"**{int(test['연도'].min())}~"
        f"{int(test['연도'].max())}년**"
    )


st.info(
    "중요: 1차·3차·9차 모델은 모두 훈련용 데이터만으로 학습했습니다. "
    "2005~2025년 테스트 데이터는 모델을 만든 뒤 성능을 평가할 때만 사용합니다."
)


# =========================================================
# 결과 표
# =========================================================

st.subheader("📊 1차·3차·9차 곡선 비교")

st.dataframe(
    results.style.format({

        "테스트 평균 오차 MAE (℃)": "{:.3f}",

        "2050년 예측기온 (℃)": "{:.2f}"

    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 가장 잘 예측한 모델
# =========================================================

best_model = results.loc[
    results["테스트 평균 오차 MAE (℃)"].idxmin(),
    "모델"
]

best_mae = results[
    results["모델"] == best_model
]["테스트 평균 오차 MAE (℃)"].iloc[0]


st.success(
    f"테스트 데이터에서 평균적으로 가장 적게 빗나간 모델은 "
    f"**{best_model}**이며, 평균 오차는 "
    f"**{best_mae:.3f}℃**입니다."
)


# =========================================================
# 테스트 데이터 실제값 vs 예측값
# =========================================================

st.subheader(
    "📈 테스트 데이터에서 실제 기온과 예측값 비교"
)

fig_test = go.Figure()


# 실제값
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        line=dict(
            width=4
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


# 1차
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_1,
        mode="lines",
        name="1차 예측",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "1차 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 3차
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_3,
        mode="lines",
        name="3차 예측",
        line=dict(
            width=3,
            dash="dot"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "3차 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 9차
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_9,
        mode="lines",
        name="9차 예측",
        line=dict(
            width=3,
            dash="longdash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "9차 예측: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig_test.update_layout(

    xaxis=dict(
        title="연도",
        type="linear",
        tickmode="linear",
        dtick=2
    ),

    yaxis=dict(
        title="연평균기온 (℃)"
    ),

    height=650,

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
# 전체 데이터와 곡선
# =========================================================

st.subheader(
    "📈 전체 기간의 실제 기온과 학습된 곡선"
)


# ---------------------------------------------------------
# 그래프용 연도
# ---------------------------------------------------------

graph_start = int(
    yearly["연도"].min()
)

graph_end = 2050

graph_years = np.linspace(
    graph_start,
    graph_end,
    700
)

graph_x = (
    graph_years - TEST_START
)

graph_X = pd.DataFrame({
    "변환연도": graph_x
})


# 예측
graph_pred_1 = model_1.predict(
    graph_X
)

graph_pred_3 = model_3.predict(
    graph_X
)

graph_pred_9 = model_9.predict(
    graph_X
)


fig_all = go.Figure()


# 실제 데이터
fig_all.add_trace(
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


# 1차
fig_all.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_1,
        mode="lines",
        name="1차 곡선",
        line=dict(
            width=3
        )
    )
)


# 3차
fig_all.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_3,
        mode="lines",
        name="3차 곡선",
        line=dict(
            width=3
        )
    )
)


# 9차
fig_all.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_9,
        mode="lines",
        name="9차 곡선",
        line=dict(
            width=3
        )
    )
)


# 테스트 시작선
fig_all.add_vline(
    x=TEST_START,
    line_width=2,
    line_dash="dash"
)


fig_all.add_annotation(
    x=TEST_START,
    y=1,
    yref="paper",
    text="테스트 시작",
    showarrow=False,
    yanchor="bottom"
)


# 2050년 선
fig_all.add_vline(
    x=2050,
    line_width=2,
    line_dash="dot"
)


fig_all.add_annotation(
    x=2050,
    y=1,
    yref="paper",
    text="2050년",
    showarrow=False,
    yanchor="bottom"
)


fig_all.update_layout(

    xaxis=dict(
        title="연도",
        type="linear",
        tickmode="linear",
        dtick=10
    ),

    yaxis=dict(
        title="연평균기온 (℃)"
    ),

    height=700,

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
    fig_all,
    use_container_width=True
)


# =========================================================
# 2050년 예측값 크게 표시
# =========================================================

st.subheader("🔮 2050년 예측 비교")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "1차 (직선)",
        f"{prediction_2050_1:.2f}℃"
    )


with col2:

    st.metric(
        "3차 곡선",
        f"{prediction_2050_3:.2f}℃"
    )


with col3:

    st.metric(
        "9차 곡선",
        f"{prediction_2050_9:.2f}℃"
    )


# =========================================================
# 주의할 점
# =========================================================

st.subheader("💡 해석할 때 주의할 점")

st.write(
    "• MAE는 테스트 기간의 실제 연평균기온과 예측값의 평균적인 차이입니다. "
    "작을수록 테스트 데이터를 더 잘 예측한 것입니다."
)

st.write(
    "• 3차와 9차 모델은 훈련 데이터의 굴곡을 더 자세하게 따라갈 수 있지만, "
    "훈련 범위를 벗어난 2050년 예측에서는 크게 휘어질 수도 있습니다."
)

st.write(
    "• 특히 9차 모델은 고차식이므로 미래로 외삽할 때 매우 큰 값이 나올 수 있습니다. "
    "그래서 연도를 그대로 사용하지 않고 `연도 - 2005`로 변환하여 계산했습니다."
)

st.write(
    "• 2050년은 테스트 기간인 2005~2025년보다 훨씬 미래이므로, "
    "2050년 예측값은 테스트 MAE와 별도로 해석해야 합니다."
)


# =========================================================
# 데이터 처리 기준
# =========================================================

st.subheader("📌 데이터 처리 기준")

st.write(
    "• 2025년 이후의 데이터는 사용하지 않았습니다."
)

st.write(
    "• 한 해의 관측일이 300일 미만인 연도는 제외했습니다."
)

st.write(
    "• 남은 일평균기온을 연도별로 평균내어 연평균기온을 계산했습니다."
)

st.write(
    "• 2005년 이전 데이터를 훈련 데이터로 사용했습니다."
)

st.write(
    "• 2005~2025년 데이터를 테스트 데이터로 사용했습니다."
)

st.write(
    "• 테스트 데이터는 어떤 모델의 학습에도 사용하지 않았습니다."
)

st.write(
    "• 모든 모델은 동일한 훈련 데이터로 비교했습니다."
)
