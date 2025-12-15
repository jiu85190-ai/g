import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------
# 1. 파일 경로 설정 (사용자 환경에 맞게 유지)
# ---------------------------------------------
FILE_PATH = "/content/drive/MyDrive/Crop_recommendation.csv"

# ---------------------------------------------
# 2. 페이지 설정
# ---------------------------------------------
st.set_page_config(
    page_title="온도 구간별 최적 작물 추천",
    layout="wide"
)

# ---------------------------------------------
# 3. 데이터 로드 및 전처리
# ---------------------------------------------
@st.cache_data
def load_data(file_path):
    """CSV 파일을 로드하고 결측치가 있는 행을 제거한 뒤, 이상치도 제거하고 DataFrame을 반환합니다."""
    try:
        df = pd.read_csv(file_path)
        
        # --- 결측치 처리 ---
        initial_rows = len(df)  # 데이터셋 로드 전의 행 수 저장
        df.dropna(inplace=True)  # 결측값(NaN)이 포함된 모든 행을 제거합니다.
        
        # 결측치 제거 후 정보 출력
        st.sidebar.info(f"데이터셋 로드 완료: {initial_rows}행 -> 결측치 제거 후 {len(df)}행")
        
        # --- 이상치 처리 (IQR 방법) ---
        # 1. 온도(temperature) 변수의 Q1, Q3 및 IQR 계산
        Q1 = df['temperature'].quantile(0.25)
        Q3 = df['temperature'].quantile(0.75)
        IQR = Q3 - Q1
        
        # 2. 이상치의 경계값(Lower & Upper Bound) 설정
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 3. 이상치 제거
        initial_rows_with_outliers = len(df)
        df = df[(df['temperature'] >= lower_bound) & (df['temperature'] <= upper_bound)]
        
        # 4. 사용자에게 정보 제공
        removed_outliers = initial_rows_with_outliers - len(df)
        if removed_outliers > 0:
            st.sidebar.warning(f"🌡️ 온도 이상치 제거: {removed_outliers}개 행 제거됨.")
        
        # --- 정규화 (Min-Max Normalization) ---
        # 정규화 대상이 되는 수치형 변수들을 목록으로 만듭니다.
        numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # 정규화 결과를 저장할 새로운 DataFrame을 만듭니다.
        df_normalized = df.copy()
        
        # 각 수치형 변수에 대해 Min-Max 정규화를 적용합니다.
        for col in numerical_features:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            
            # 정규화 공식: (현재 값 - 최소값) / (최대값 - 최소값)
            df_normalized[col + '_normalized'] = (df_normalized[col] - min_val) / (max_val - min_val)

        # st.sidebar.info("데이터의 수치형 변수들이 0-1 범위로 정규화되었습니다.") 

        return df_normalized  # 정규화된 데이터프레임 반환
    except FileNotFoundError:
        st.error(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None

# ---------------------------------------------
# 4. Streamlit UI (사용자 인터페이스)
# ---------------------------------------------
st.title("🌱 온도 구간별 최적 작물 추천 분석")
st.markdown("전체 작물 빈도 대신, **각 온도 구간별로 추천 빈도가 가장 높은(최적) 작물 하나**만 분석하여 추천합니다.")

df = load_data(FILE_PATH)

if df is not None:

    # -------------------
    # 온도 구간 설정 (사용자 지정 가능하게 사이드바 추가)
    # -------------------
    st.sidebar.header("⚙️ 분석 설정")

    # 구간 개수 설정 슬라이더
    num_bins = st.sidebar.slider("온도 구간(Bins) 개수 선택", 3, 15, 5)

    # 온도 변수를 선택된 구간 개수로 나누어 새로운 카테고리 컬럼 생성
    bin_labels = [f'Bin {i+1}' for i in range(num_bins)]
    df['temp_bin'] = pd.cut(
        df['temperature'],
        bins=num_bins,
        include_lowest=True,
        labels=bin_labels
    )

    # -------------------
    # 최적 작물 데이터 집계
    # -------------------
    # 1. 각 온도 구간(temp_bin) 및 작물(label)별 빈도수 계산
    grouped_counts = df.groupby(['temp_bin', 'label']).size().reset_index(name='count')

    # 2. 각 온도 구간(temp_bin) 내에서 'count'가 최대인 행(작물)만 추출
    # idxmax()를 사용하여 최대값의 인덱스를 찾고, loc[]로 해당 행을 선택
    best_crop_per_bin = grouped_counts.loc[grouped_counts.groupby('temp_bin')['count'].idxmax()]

    # 결과 DataFrame 정리 및 컬럼 이름 변경
    best_crop_per_bin = best_crop_per_bin.rename(columns={'label': '최적 작물', 'count': '최대 추천 빈도수'})
    best_crop_per_bin = best_crop_per_bin.reset_index(drop=True)

    # temp_bin을 원래 순서대로 정렬 (Bin 1, Bin 2, ...)
    best_crop_per_bin['temp_bin'] = pd.Categorical(
        best_crop_per_bin['temp_bin'],
        categories=bin_labels,
        ordered=True
    )
    best_crop_per_bin = best_crop_per_bin.sort_values('temp_bin')

    # -------------------
    # 시각화 및 결과 테이블
    # -------------------
    st.header(f"🌡️ 온도 {num_bins}개 구간별 최적 작물 추천 결과")
    st.subheader("✅ 구간별 최고 추천 작물 (최적 작물)")

    # 테이블 출력
    st.dataframe(
        best_crop_per_bin[['temp_bin', '최적 작물', '최대 추천 빈도수']],
        use_container_width=True,
        hide_index=True
    )

    # 시각화 (막대 그래프)
    # 온도 구간별 최고 추천 작물의 빈도수를 시각화하여 비교
    fig_best = px.bar(
        best_crop_per_bin,
        x='temp_bin',
        y='최대 추천 빈도수',
        color='최적 작물',  # 최적 작물 종류별로 색상 구분
        text='최적 작물',  # 막대 위에 최적 작물 이름 표시
        title=f"온도 구간별 최고 추천 작물 빈도",
        labels={'temp_bin': '온도 구간', '최대 추천 빈도수': '최대 추천 빈도수', '최적 작물': '최적 작물'},
        height=650
    )

    # 텍스트 레이블 설정
    fig_best.update_traces(textposition='outside')
    fig_best.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    # X축 순서를 구간 순서대로 정렬
    fig_best.update_xaxes(categoryorder="array", categoryarray=bin_labels)

    st.plotly_chart(fig_best, use_container_width=True)

    st.markdown(
        """
        ### 분석 해석 가이드:
        - **테이블/막대 그래프**: 각 온도 구간(`temp_bin`)에서 데이터셋에 의해 **가장 많이 추천된** 작물(`최적 작물`)을 보여줍니다.
        - **최대 추천 빈도수**: 해당 작물이 그 온도 구간에서 추천된 횟수로, 잠재적 적합도를 나타내는 간접 지표입니다.
        - **활용**: 이 결과를 통해 특정 온도 조건에 가장 잘 맞는 것으로 예측되는 작물을 빠르게 파악할 수 있습니다.
        """
    )

# ---------------------------------------------
# 5. 구현 코드 섹션 추가
# ---------------------------------------------
st.subheader("📜 구현 코드")

# 코드만 보여주는 칸을 만들기 위해 `st.code()` 사용
st.code("""
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
        initial_rows = len(df)
        df.dropna(inplace=True)
        st.sidebar.info(f"데이터셋 로드 완료: {initial_rows}행 -> 결측치 제거 후 {len(df)}행")
        
        Q1 = df['temperature'].quantile(0.25)
        Q3 = df['temperature'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['temperature'] >= lower_bound) & (df['temperature'] <= upper_bound)]
        
        numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        df_normalized = df.copy()
        for col in numerical_features:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            df_normalized[col + '_normalized'] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        return df_normalized
    except FileNotFoundError:
        st.error(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None

st.title("🌱 온도 구간별 최적 작물 추천 분석")
st.markdown("전체 작물 빈도 대신, 각 온도 구간별로 추천 빈도가 가장 높은 작물 하나만 분석합니다.")

df = load_data(FILE_PATH)

if df is not None:
    st.sidebar.header("⚙️ 분석 설정")
    num_bins = st.sidebar.slider("온도 구간(Bins) 개수 선택", 3, 15, 5)
    bin_labels = [f'Bin {i+1}' for i in range(num_bins)]
    df['temp_bin'] = pd.cut(df['temperature'], bins=num_bins, include_lowest=True, labels=bin_labels)
    
    grouped_counts = df.groupby(['temp_bin', 'label']).size().reset_index(name='count')
    best_crop_per_bin = grouped_counts.loc[grouped_counts.groupby('temp_bin')['count'].idxmax()]
    best_crop_per_bin = best_crop_per_bin.rename(columns={'label': '최적 작물', 'count': '최대 추천 빈도수'}).reset_index(drop=True)
    best_crop_per_bin['temp_bin'] = pd.Categorical(best_crop_per_bin['temp_bin'], categories=bin_labels, ordered=True)
    best_crop_per_bin = best_crop_per_bin.sort_values('temp_bin')
    
    st.header(f"🌡️ 온도 {num_bins}개 구간별 최적 작물 추천 결과")
    st.subheader("✅ 구간별 최고 추천 작물")
    st.dataframe(best_crop_per_bin[['temp_bin', '최적 작물', '최대 추천 빈도수']], use_container_width=True, hide_index=True)
    
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
""")
