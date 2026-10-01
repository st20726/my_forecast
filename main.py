import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온 데이터를 이용해 미래의 기온을 예측합니다.")

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 결측값 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 2025년까지만 사용
# --------------------------------------------------
df = df[df["연도"] <= 2025].copy()


# --------------------------------------------------
# 연도별 관측일 수와 연평균기온 계산
# --------------------------------------------------
yearly = (
    df.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)

# 관측일이 300일 미만인 해 제외
yearly = yearly[yearly["관측일수"] >= 300].copy()

# 연도순 정렬
yearly = yearly.sort_values("연도").reset_index(drop=True)


# --------------------------------------------------
# 회귀에 사용할 데이터 확인
# --------------------------------------------------
if len(yearly) < 2:
    st.error("회귀분석을 수행할 수 있는 연도 데이터가 충분하지 않습니다.")
    st.stop()


# --------------------------------------------------
# 독립변수:
# 1908년부터 지난 연수
#
# 1908년 → 0
# 1909년 → 1
# 1910년 → 2
# ...
# --------------------------------------------------
yearly["지난연수"] = yearly["연도"] - 1908


# --------------------------------------------------
# 선형회귀 계산
# y = a*x + b
# --------------------------------------------------
x = yearly["지난연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)

# 예측값
yearly["회귀예측기온"] = slope * yearly["지난연수"] + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# --------------------------------------------------
# 회귀선 범위
# --------------------------------------------------
regression_start_year = int(yearly["연도"].min())
regression_end_year = int(yearly["연도"].max())
number_of_years = len(yearly)


# --------------------------------------------------
# 회귀선의 표시 범위
# 실제 데이터가 있는 구간에서 표시
# --------------------------------------------------
line_years = np.linspace(
    regression_start_year,
    regression_end_year,
    300
)

line_x = line_years - 1908
line_y = slope * line_x + intercept


# --------------------------------------------------
# 주요 정보 표시
# --------------------------------------------------
st.subheader("📊 회귀분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{number_of_years}년")

with col2:
    st.metric("시작 연도", f"{regression_start_year}년")

with col3:
    st.metric("끝 연도", f"{regression_end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")


st.caption(
    f"2025년까지의 데이터 중 연간 관측일이 300일 이상인 {number_of_years}개 연도를 사용했습니다."
)


# --------------------------------------------------
# 산점도 + 회귀선
# --------------------------------------------------
st.subheader("📈 연도별 연평균기온과 회귀선")

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 예측기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis=dict(
        title="연도",
        type="linear",
        tickmode="linear",
        dtick=10
    ),
    yaxis=dict(
        title="연평균기온 (℃)"
    ),
    hovermode="closest",
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 상관계수 설명
# --------------------------------------------------
st.subheader("🔎 상관계수")

st.write(
    f"연도와 연평균기온의 상관계수는 **{correlation:.3f}**입니다."
)

if correlation > 0:
    st.write(
        "상관계수가 양수이므로 이 데이터에서는 연도가 증가할수록 "
        "연평균기온도 높아지는 경향이 나타납니다."
    )
elif correlation < 0:
    st.write(
        "상관계수가 음수이므로 이 데이터에서는 연도가 증가할수록 "
        "연평균기온이 낮아지는 경향이 나타납니다."
    )
else:
    st.write(
        "상관계수가 0에 가까워 연도와 연평균기온 사이의 "
        "선형적인 관계가 매우 약합니다."
    )


# --------------------------------------------------
# 연도 슬라이더
# --------------------------------------------------
st.subheader("🌡️ 원하는 연도의 예상 기온")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 선택한 연도의 회귀 예측값
selected_x = selected_year - 1908
predicted_temperature = slope * selected_x + intercept


# --------------------------------------------------
# 예상 기온 크게 표시
# --------------------------------------------------
st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 30px;
        margin: 10px 0 30px 0;
        border-radius: 15px;
        background-color: #f0f4f8;
    ">
        <div style="font-size: 24px; margin-bottom: 10px;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="font-size: 56px; font-weight: bold;">
            {predicted_temperature:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 선택한 연도를 그래프에 표시
# --------------------------------------------------
st.subheader("📍 선택한 연도의 위치")

fig_selected = go.Figure()

# 회귀선
fig_selected.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="회귀선",
        line=dict(width=3)
    )
)

# 선택한 연도
fig_selected.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(
            size=15,
            symbol="circle"
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            f"예상 연평균기온: {predicted_temperature:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig_selected.update_layout(
    xaxis=dict(
        title="연도",
        range=[1900, 2100],
        dtick=10
    ),
    yaxis=dict(
        title="연평균기온 (℃)"
    ),
    height=450,
    hovermode="closest"
)

st.plotly_chart(fig_selected, use_container_width=True)


# --------------------------------------------------
# 데이터 조건 안내
# --------------------------------------------------
st.subheader("📌 데이터 처리 기준")

st.write(
    "• 2025년 이후의 데이터는 분석에서 제외했습니다."
)

st.write(
    "• 한 해의 관측일이 300일 미만인 연도는 분석에서 제외했습니다."
)

st.write(
    "• 남은 각 연도의 일평균기온을 평균내어 연평균기온을 계산했습니다."
)

st.write(
    "• 회귀분석의 독립변수는 `연도 - 1908`로 설정했습니다."
)

st.write(
    "• 따라서 1908년은 독립변수 0, 1909년은 1, 1910년은 2가 됩니다."
)
