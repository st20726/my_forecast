import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온을 이용해 장기적인 기온 변화를 살펴봅니다.")


# --------------------------------------------------
# 데이터 주소
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
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

    # 필요한 데이터만 남기기
    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 연도 만들기
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


# --------------------------------------------------
# 관측일 300일 미만인 해 제외
# --------------------------------------------------
yearly = yearly[
    yearly["관측일수"] >= 300
].copy()


yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# --------------------------------------------------
# 데이터가 충분한지 확인
# --------------------------------------------------
if len(yearly) < 2:
    st.error("회귀분석을 수행할 데이터가 충분하지 않습니다.")
    st.stop()


# --------------------------------------------------
# 독립변수
#
# 1908년 → 0
# 1909년 → 1
# 1910년 → 2
# ...
# --------------------------------------------------
yearly["지난연수"] = yearly["연도"] - 1908


# ==================================================
# 1. 전체 기간 회귀
# ==================================================

x_all = yearly["지난연수"].to_numpy()
y_all = yearly["연평균기온"].to_numpy()

slope_all, intercept_all = np.polyfit(
    x_all,
    y_all,
    1
)

# 1년에 몇 도 변화하는지
slope_all_per_year = slope_all

# 100년에 몇 도 변화하는지
slope_all_per_100 = slope_all * 100


# --------------------------------------------------
# 전체 기간 상관계수
# --------------------------------------------------
correlation_all = np.corrcoef(
    x_all,
    y_all
)[0, 1]


# ==================================================
# 2. 최근 20년 회귀
# ==================================================
# 기준 기간이 2025년까지이므로
# 최근 20년 = 2006~2025
# ==================================================

recent_start = 2025 - 19
recent_end = 2025

recent = yearly[
    (yearly["연도"] >= recent_start)
    & (yearly["연도"] <= recent_end)
].copy()


if len(recent) >= 2:

    x_recent = recent["지난연수"].to_numpy()
    y_recent = recent["연평균기온"].to_numpy()

    slope_recent, intercept_recent = np.polyfit(
        x_recent,
        y_recent,
        1
    )

    slope_recent_per_year = slope_recent
    slope_recent_per_100 = slope_recent * 100

    correlation_recent = np.corrcoef(
        x_recent,
        y_recent
    )[0, 1]

else:

    slope_recent = np.nan
    intercept_recent = np.nan
    slope_recent_per_year = np.nan
    slope_recent_per_100 = np.nan
    correlation_recent = np.nan


# --------------------------------------------------
# 회귀 기간 정보
# --------------------------------------------------
regression_start_year = int(
    yearly["연도"].min()
)

regression_end_year = int(
    yearly["연도"].max()
)

number_of_years = len(yearly)


# ==================================================
# 3. 회귀선용 데이터
# ==================================================

line_years_all = np.linspace(
    regression_start_year,
    regression_end_year,
    400
)

line_x_all = line_years_all - 1908

line_y_all = (
    slope_all * line_x_all
    + intercept_all
)


# 최근 20년 회귀선
line_years_recent = np.linspace(
    recent_start,
    recent_end,
    200
)

line_x_recent = line_years_recent - 1908

line_y_recent = (
    slope_recent * line_x_recent
    + intercept_recent
)


# ==================================================
# 4. 가장 중요한 기울기 표시
# ==================================================

st.subheader("🌡️ 100년에 몇 ℃ 오르는가?")

col1, col2 = st.columns(2)


with col1:

    st.markdown("### 전체 기간")

    st.metric(
        label="100년에 변화하는 연평균기온",
        value=f"{slope_all_per_100:+.2f} ℃"
    )

    st.caption(
        f"{regression_start_year}~{regression_end_year}년 "
        f"{number_of_years}개 연도 사용"
    )


with col2:

    st.markdown("### 최근 20년")

    st.metric(
        label="100년에 변화하는 연평균기온",
        value=f"{slope_recent_per_100:+.2f} ℃"
    )

    st.caption(
        f"{recent_start}~{recent_end}년 "
        f"{len(recent)}개 연도 사용"
    )


# ==================================================
# 5. 전체 기간 vs 최근 20년 비교
# ==================================================

st.subheader("📊 전체 기간과 최근 20년 비교")

comparison = pd.DataFrame({
    "구분": [
        "전체 기간",
        "최근 20년"
    ],
    "기간": [
        f"{regression_start_year}~{regression_end_year}",
        f"{recent_start}~{recent_end}"
    ],
    "사용 연도 수": [
        number_of_years,
        len(recent)
    ],
    "100년당 기온 변화(℃)": [
        slope_all_per_100,
        slope_recent_per_100
    ],
    "상관계수": [
        correlation_all,
        correlation_recent
    ]
})

st.dataframe(
    comparison.style.format({
        "100년당 기온 변화(℃)": "{:+.2f}",
        "상관계수": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 6. 산점도 + 회귀선
# ==================================================

st.subheader("📈 연도별 연평균기온과 회귀선")

fig = go.Figure()


# --------------------------------------------------
# 실제 연평균기온
# --------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        customdata=yearly[
            ["관측일수"]
        ].to_numpy(),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 전체 기간 회귀선
# --------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=line_years_all,
        y=line_y_all,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=4
        ),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 예측기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 최근 20년 회귀선
# --------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=line_years_recent,
        y=line_y_recent,
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(
            width=4,
            dash="dash"
        ),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "최근 20년 회귀 예측기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 그래프 설정
# --------------------------------------------------
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
    fig,
    use_container_width=True
)


# ==================================================
# 7. 상관계수
# ==================================================

st.subheader("🔎 상관계수")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 전체 기간")

    st.metric(
        "상관계수",
        f"{correlation_all:.3f}"
    )


with col2:

    st.markdown("### 최근 20년")

    st.metric(
        "상관계수",
        f"{correlation_recent:.3f}"
    )


# ==================================================
# 8. 연도 슬라이더
# ==================================================

st.subheader("🌡️ 원하는 연도의 예상 기온")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# --------------------------------------------------
# 전체 기간 회귀식을 이용한 예상 기온
# --------------------------------------------------
selected_x = selected_year - 1908

predicted_temperature = (
    slope_all * selected_x
    + intercept_all
)


# --------------------------------------------------
# 큰 숫자로 표시
# --------------------------------------------------
st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 35px;
        margin: 15px 0 30px 0;
        border-radius: 18px;
        background-color: #f0f4f8;
    ">
        <div style="
            font-size: 24px;
            margin-bottom: 10px;
        ">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size: 58px;
            font-weight: bold;
        ">
            {predicted_temperature:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 9. 선택한 연도 위치 표시
# ==================================================

st.subheader("📍 선택한 연도의 위치")

fig_selected = go.Figure()


# 전체 회귀선
fig_selected.add_trace(
    go.Scatter(
        x=line_years_all,
        y=line_y_all,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=3
        )
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
            size=16
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            f"예상 연평균기온: "
            f"{predicted_temperature:.2f}℃"
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

st.plotly_chart(
    fig_selected,
    use_container_width=True
)


# ==================================================
# 10. 데이터 처리 기준
# ==================================================

st.subheader("📌 데이터 처리 기준")

st.write(
    "• 2025년 이후의 데이터는 분석에서 제외했습니다."
)

st.write(
    "• 한 해의 관측일이 300일 미만인 연도는 제외했습니다."
)

st.write(
    "• 각 연도의 일평균기온을 평균내어 연평균기온을 계산했습니다."
)

st.write(
    "• 전체 기간 회귀에서는 독립변수를 `연도 - 1908`로 설정했습니다."
)

st.write(
    f"• 최근 20년 회귀에서는 {recent_start}~{recent_end}년의 데이터만 사용했습니다."
)

st.write(
    "• 회귀 기울기는 이해하기 쉽도록 '1년당 ℃ 변화량 × 100'으로 바꾸어 "
    "'100년에 몇 ℃ 변화하는가'로 표시했습니다."
)
