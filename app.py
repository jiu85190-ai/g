import streamlit as st
import pandas as pd
import plotly.express as px

FILE_PATH = "/content/drive/MyDrive/Crop_recommendation.csv"

st.set_page_config(
    page_title="온도 구간별 최적 작물 추천",
    layout="wide"
)

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None

st.title("🌱 온도 구간별 최적 작물 추천 분석")
st.markdown("전체 작물 빈도 대신, **각 온도 구간별로 추천 빈도가 가장 높은(최적) 작물 하나**만 분석하여 추천합니다.")

df = load_data(FILE_PATH)

if df is not None:
    st.sidebar.header("⚙️ 분석 설정")

    num_bins = st.sidebar.slider("온도 구간(Bins) 개수 선택", 3, 15, 5)

    bin_labels = [f'Bin {i+1}' for i in range(num_bins)]
    df['temp_bin'] = pd.cut(
        df['temperature'],
        bins=num_bins,
        include_lowest=True,
        labels=bin_labels
    )

    grouped_counts = df.groupby(['temp_bin', 'label']).size().reset_index(name='count')

    best_crop_per_bin = grouped_counts.loc[grouped_counts.groupby('temp_bin')['count'].idxmax()]

    best_crop_per_bin = best_crop_per_bin.rename(columns={'label': '최적 작물', 'count': '최대 추천 빈도수'})
    best_crop_per_bin = best_crop_per_bin.reset_index(drop=True)

    best_crop_per_bin['temp_bin'] = pd.Categorical(
        best_crop_per_bin['temp_bin'],
        categories=bin_labels,
        ordered=True
    )
    best_crop_per_bin = best_crop_per_bin.sort_values('temp_bin')

    st.header(f"🌡️ 온도 {num_bins}개 구간별 최적 작물 추천 결과")
    st.subheader("✅ 구간별 최고 추천 작물 (최적 작물)")

    st.dataframe(
        best_crop_per_bin[['temp_bin', '최적 작물', '최대 추천 빈도수']],
        use_container_width=True,
        hide_index=True
    )

    fig_best = px.bar(
        best_crop_per_bin,
        x='temp_bin',
        y='최대 추천 빈도수',
        color='최적 작물',
        text='최적 작물',
        title=f"온도 구간별 최고 추천 작물 빈도",
        labels={'temp_bin': '온도 구간', '최대 추천 빈도수': '최대 추천 빈도수', '최적 작물': '최적 작물'},
        height=650
    )

    fig_best.update_traces(textposition='outside')
    fig_best.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    fig_best.update_xaxes(categoryorder="array", categoryarray=bin_labels)

    st.plotly_chart(fig_best, use_container_width=True)