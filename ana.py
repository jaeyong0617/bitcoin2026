import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.linear_model import LinearRegression

# 페이지 설정
st.set_page_config(page_title="Bitcoin Analysis Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_candidates = ["coin.csv", "coin.csv.csv"]
    file_path = None
    
    for candidate in file_candidates:
        if os.path.exists(candidate):
            file_path = candidate
            break
            
    if file_path is None:
        return None
    
    try:
        df = pd.read_csv(file_path, delimiter=';')
        df['timeOpen'] = pd.to_datetime(df['timeOpen'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timeOpen')
        
        # 기본 지표 계산
        df['daily_return'] = df['close'].pct_change() * 100
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        
        return df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def predict_next_day(df):
    """선형회귀를 이용한 내일 가격 예측"""
    # 학습을 위한 데이터 준비 (최근 60일 데이터 사용)
    data = df[['close']].copy()
    window_size = 5 # 과거 5일 데이터를 보고 다음날을 예측
    
    for i in range(1, window_size + 1):
        data[f'shift_{i}'] = data['close'].shift(i)
    
    data = data.dropna()
    
    X = data[[f'shift_{i}' for i in range(1, window_size + 1)]]
    y = data['close']
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 마지막 데이터를 바탕으로 내일 가격 예측
    last_data = df['close'].iloc[-window_size:].values[::-1].reshape(1, -1)
    prediction = model.predict(last_data)[0]
    
    return prediction

# 대시보드 메인 로직
try:
    df = load_data()

    if df is None:
        st.error("⚠️ 데이터를 불러올 수 없습니다.")
        st.info("파이썬 파일과 동일한 폴더에 `coin.csv` 파일이 있는지 확인해주세요.")
    else:
        # 사이드바 설정
        st.sidebar.header("📊 분석 및 예측")
        
        # 날짜 범위 선택
        min_date = df['timeOpen'].min().date()
        max_date = df['timeOpen'].max().date()
        date_range = st.sidebar.date_input("분석 기간 선택", value=(min_date, max_date))
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df['timeOpen'].dt.date >= start_date) & (df['timeOpen'].dt.date <= end_date)
            filtered_df = df.loc[mask]
        else:
            filtered_df = df

        st.title("₿ 비트코인 가격 분석 및 AI 예측")

        # --- 예측 섹션 ---
        st.subheader("🚀 AI 내일 가격 예측 (Linear Regression)")
        
        # 전체 데이터를 기반으로 학습 후 예측
        predicted_price = predict_next_day(df)
        current_price = df['close'].iloc[-1]
        change = predicted_price - current_price
        change_pct = (change / current_price) * 100
        
        p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
        
        with p_col1:
            st.metric("현재 가격 (오늘)", f"₩{current_price:,.0f}")
        with p_col2:
            color = "normal" if change >= 0 else "inverse"
            st.metric("내일 예상 가격", f"₩{predicted_price:,.0f}", f"{change_pct:+.2f}%", delta_color=color)
        
        with p_col3:
            if change > 0:
                st.success(f"📈 분석 결과: 내일은 약 **{change_pct:.2f}% 상승**할 것으로 예측됩니다. (매수 고려 가능)")
            else:
                st.error(f"📉 분석 결과: 내일은 약 **{abs(change_pct):.2f}% 하락**할 것으로 예측됩니다. (주의 필요)")
        
        st.info("💡 예측 모델 안내: 과거 5일간의 종가 패턴을 선형 회귀로 분석한 결과입니다. 투자의 책임은 본인에게 있습니다.")
        st.divider()

        # --- 지표 섹션 ---
        col1, col2, col3, col4 = st.columns(4)
        latest_data = filtered_df.iloc[-1]
        prev_data = filtered_df.iloc[-2] if len(filtered_df) > 1 else latest_data
        
        price_diff = latest_data['close'] - prev_data['close']
        price_pct = (price_diff / prev_data['close']) * 100 if prev_data['close'] != 0 else 0
        
        col1.metric("현재 종가", f"₩{latest_data['close']:,.0f}", f"{price_pct:.2f}%")
        col2.metric("기간 내 최고가", f"₩{filtered_df['high'].max():,.0f}")
        col3.metric("기간 내 최저가", f"₩{filtered_df['low'].min():,.0f}")
        col4.metric("최근 거래량", f"{latest_data['volume']:,.0e}")

        # --- 차트 섹션 ---
        st.subheader("캔들스틱 및 기술적 지표")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                            subplot_titles=('Price Trend', 'Volume'), row_width=[0.3, 0.7])

        fig.add_trace(go.Candlestick(
            x=filtered_df['timeOpen'], open=filtered_df['open'], high=filtered_df['high'],
            low=filtered_df['low'], close=filtered_df['close'], name="시가/고가/저가/종가"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=filtered_df['timeOpen'], y=filtered_df['MA20'], name="20일 이평선", 
                                 line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_df['timeOpen'], y=filtered_df['MA50'], name="50일 이평선", 
                                 line=dict(color='blue', width=1)), row=1, col=1)

        colors = ['red' if row['close'] < row['open'] else 'green' for _, row in filtered_df.iterrows()]
        fig.add_trace(go.Bar(x=filtered_df['timeOpen'], y=filtered_df['volume'], name="거래량", 
                             marker_color=colors, opacity=0.6), row=2, col=1)

        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # --- 상세 데이터 ---
        with st.expander("데이터 통계 및 원본 보기"):
            st.table(filtered_df[['open', 'high', 'low', 'close', 'volume']].describe().T.style.format("{:,.0f}"))
            st.dataframe(filtered_df.sort_values('timeOpen', ascending=False))

except Exception as e:
    st.error(f"오류가 발생했습니다.")
    st.exception(e)
