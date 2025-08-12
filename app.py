# app.py

# საჭირო ბიბლიოთეკების იმპორტი
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np
import warnings
import pickle
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Custom ტექნიკური ინდიკატორების ფუნქციები
def calculate_ema(data, period):
    """EMA (Exponential Moving Average) გამოთვლა"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """RSI (Relative Strength Index) გამოთვლა"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence) გამოთვლა"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    macd_signal = calculate_ema(macd, signal)
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Bollinger Bands გამოთვლა"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def save_forecast(ticker, forecast_data, model_date):
    """პროგნოზის შენახვა ფაილში"""
    try:
        if not os.path.exists('forecasts'):
            os.makedirs('forecasts')
        
        forecast_file = f'forecasts/{ticker}_forecast.pkl'
        forecast_dict = {
            'forecast': forecast_data,
            'model_date': model_date,
            'ticker': ticker
        }
        
        with open(forecast_file, 'wb') as f:
            pickle.dump(forecast_dict, f)
        
        return True
    except Exception as e:
        st.warning(f"პროგნოზის შენახვის შეცდომა: {e}")
        return False

def load_previous_forecast(ticker):
    """წინა პროგნოზის ჩატვირთვა"""
    try:
        forecast_file = f'forecasts/{ticker}_forecast.pkl'
        if os.path.exists(forecast_file):
            with open(forecast_file, 'rb') as f:
                forecast_dict = pickle.load(f)
            return forecast_dict
        return None
    except Exception as e:
        st.warning(f"წინა პროგნოზის ჩატვირთვის შეცდომა: {e}")
        return None

# Streamlit-ის გვერდის კონფიგურაცია
st.set_page_config(layout="wide", page_title="კრიპტოვალუტის პროგნოზირების აპლიკაცია")

# --- UI ინტერფეისის ნაწილი ---
st.title("🚀 კრიპტოვალუტების ფასის პროგნოზი და სიგნალები")
st.markdown("ეს აპლიკაცია იყენებს ისტორიულ მონაცემებს პროგნოზირებისა და ტრეიდინგის სიგნალების გენერირებისთვის.")

# მომხმარებლისთვის კრიპტოვალუტის და ინტერვალის ასარჩევად
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.selectbox(
        "📈 აირჩიეთ კრიპტოვალუტა",
        ("BTC-USD", "ETH-USD", "XRP-USD", "SAND-USD", "ADA-USD", "SOL-USD", "SKL-USD", "RVN-USD")
    )
with col2:
    interval = st.selectbox(
        "⏰ აირჩიეთ დროის ინტერვალი",
        ("1d", "1h", "4h", "15m")
    )
with col3:
    forecast_days = st.slider("📊 პროგნოზის დღეები", min_value=7, max_value=90, value=30)

# მონაცემების კეშირება, რომ ყოველ ჯერზე არ მოხდეს გადმოწერა
@st.cache_data(ttl=300)  # 5 წუთიანი TTL
def load_data(ticker, interval):
    """
    yfinance-დან მონაცემების გადმოწერა არჩეული კრიპტოსა და ინტერვალისთვის.
    """
    try:
        # განსხვავებული ინტერვალებისთვის განსხვავებული პერიოდები
        period_mapping = {
            "15m": "7d",   # 15 წუთიანი მონაცემები მხოლოდ 7 დღისთვის
            "1h": "30d",   # 1 საათიანი მონაცემები 30 დღისთვის  
            "4h": "90d",   # 4 საათიანი მონაცემები 90 დღისთვის
            "1d": "2y"     # დღიური მონაცემები 2 წლისთვის
        }
        
        period = period_mapping.get(interval, "1y")
        data = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            st.error(f"მონაცემები ვერ მოიძებნა ticker-ისთვის: {ticker} interval-ზე: {interval}")
            return None
            
        # MultiIndex-ის მოშორება თუ არსებობს
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Index-ის რესეტი და Date column-ის შექმნა
        data = data.reset_index()
        if 'Datetime' in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})
        
        # Date column-ის datetime-ში კონვერტაცია
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
            
        return data
        
    except Exception as e:
        st.error(f"შეცდომა მონაცემების გადმოწერისას: {e}")
        return None

# Loading spinner-ის დამატება
with st.spinner(f'{ticker} მონაცემების ჩატვირთვა...'):
    data = load_data(ticker, interval)

if data is not None and not data.empty:
    st.success(f"✅ წარმატებით ჩაიტვირთა {len(data)} მონაცემთა ჩანაწერი")
    
    # მონაცემების მომზადება სანთლის გრაფიკისთვის
    data_candle = data.copy()
    
    # NaN მნიშვნელობების შემოწმება
    if data_candle['Close'].isnull().all():
        st.error("ყველა Close ფასი არის NaN")
        st.stop()
    
    # NaN მნიშვნელობების მოშორება
    data_candle = data_candle.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
    
    if len(data_candle) < 50:
        st.warning(f"მეტისმეტად მცირე მონაცემები ტექნიკური ანალიზისთვის: {len(data_candle)} ჩანაწერი")
    
    # ტექნიკური ინდიკატორების გამოთვლა
    try:
        # EMA (მოძრავი საშუალო) გამოთვლა
        data_candle['EMA_9'] = calculate_ema(data_candle['Close'], 9)
        data_candle['EMA_21'] = calculate_ema(data_candle['Close'], 21)
        data_candle['EMA_50'] = calculate_ema(data_candle['Close'], 50)
        
        # RSI (Relative Strength Index) გამოთვლა
        data_candle['RSI'] = calculate_rsi(data_candle['Close'], 14)
        
        # MACD (Moving Average Convergence Divergence) გამოთვლა
        macd, macd_signal, macd_histogram = calculate_macd(data_candle['Close'])
        data_candle['MACD'] = macd
        data_candle['MACD_Signal'] = macd_signal
        data_candle['MACD_Histogram'] = macd_histogram

        # Bollinger Bands-ის დამატება
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data_candle['Close'])
        data_candle['BB_Upper'] = bb_upper
        data_candle['BB_Middle'] = bb_middle
        data_candle['BB_Lower'] = bb_lower
            
    except Exception as e:
        st.warning(f"ტექნიკური ინდიკატორების გამოთვლის შეცდომა: {e}")

    # --- წინა პროგნოზის ჩატვირთვა და ანალიზი ---
    previous_forecast = load_previous_forecast(ticker) if interval == '1d' else None
    
    # --- გაერთიანებული გრაფიკი ისტორიული მონაცემებითა და პროგნოზით ---
    if interval == '1d':
        st.subheader(f"🔮 {ticker} - ისტორიული მონაცემები და პროგნოზი")
        
        try:
            # Prophet-ისთვის მონაცემების მომზადება
            df_prophet = data.reset_index().copy()
            df_prophet = df_prophet[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            # NaN მნიშვნელობების მოშორება
            df_prophet = df_prophet.dropna()
            
            # ნულოვანი ან უარყოფითი ფასების მოშორება
            df_prophet = df_prophet[df_prophet['y'] > 0]
            
            if len(df_prophet) >= 30:
                # Prophet მოდელის კონფიგურაცია
                m = Prophet(
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10,
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True if len(df_prophet) > 365 else False,
                    interval_width=0.8
                )
                
                with st.spinner('პროგნოზის გაკეთება...'):
                    m.fit(df_prophet)
                    
                    # მომავალი მონაცემთა ფრეიმის შექმნა
                    future = m.make_future_dataframe(periods=forecast_days, freq='D')
                    
                    # პროგნოზის გაკეთება
                    forecast = m.predict(future)
                    
                    # ახალი პროგნოზის შენახვა
                    save_forecast(ticker, forecast, datetime.now())

                # გაერთიანებული გრაფიკის შექმნა
                fig_combined = make_subplots(
                    rows=5, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
                    subplot_titles=(
                        f'{ticker} ისტორიული მონაცემები და პროგნოზი',
                        'მოცულობა', 
                        'RSI', 
                        'MACD',
                        'პროგნოზის ზუსტობა'
                    )
                )

                # ისტორიული სანთლის გრაფიკი
                fig_combined.add_trace(go.Candlestick(
                    x=data_candle.index,
                    open=data_candle['Open'], 
                    high=data_candle['High'],
                    low=data_candle['Low'], 
                    close=data_candle['Close'],
                    name="ისტორიული ფასები",
                    showlegend=True
                ), row=1, col=1)

                # EMA ხაზების დამატება
                if 'EMA_21' in data_candle.columns:
                    fig_combined.add_trace(go.Scatter(
                        x=data_candle.index, 
                        y=data_candle['EMA_21'],
                        line=dict(color='purple', width=1), 
                        name='EMA 21'
                    ), row=1, col=1)

                # პროგნოზირებული ფასების დამატება
                forecast_dates = pd.to_datetime(forecast['ds'])
                
                # მიმდინარე თარიღი
                current_date = data_candle.index[-1]
                
                # მხოლოდ მომავალი პროგნოზი
                future_forecast = forecast[forecast['ds'] > current_date]
                
                if not future_forecast.empty:
                    # პროგნოზირებული ფასი (ძირითადი ხაზი)
                    fig_combined.add_trace(go.Scatter(
                        x=pd.to_datetime(future_forecast['ds']),
                        y=future_forecast['yhat'],
                        line=dict(color='yellow', width=3, dash='dash'),
                        name='პროგნოზი',
                        showlegend=True
                    ), row=1, col=1)
                    
                    # confidence interval-ი
                    fig_combined.add_trace(go.Scatter(
                        x=pd.to_datetime(future_forecast['ds']),
                        y=future_forecast['yhat_upper'],
                        line=dict(color='rgba(255,255,0,0.2)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=1, col=1)
                    
                    fig_combined.add_trace(go.Scatter(
                        x=pd.to_datetime(future_forecast['ds']),
                        y=future_forecast['yhat_lower'],
                        line=dict(color='rgba(255,255,0,0.2)'),
                        fill='tonexty',
                        fillcolor='rgba(255,255,0,0.1)',
                        name='პროგნოზის საზღვრები',
                        showlegend=True,
                        hoverinfo='skip'
                    ), row=1, col=1)

                # წინა პროგნოზის ანალიზი და ჩვენება
                if previous_forecast:
                    try:
                        prev_forecast_data = previous_forecast['forecast']
                        prev_model_date = previous_forecast['model_date']
                        
                        # წინა პროგნოზიდან მხოლოდ ის მონაცემები, რომელსაც რეალური მონაცემები აქვს
                        prev_forecast_df = pd.DataFrame(prev_forecast_data)
                        prev_forecast_df['ds'] = pd.to_datetime(prev_forecast_df['ds'])
                        
                        # მხოლოდ ის წინა პროგნოზი, რომელიც ახლა რეალურია
                        historical_period = prev_forecast_df[
                            (prev_forecast_df['ds'] >= prev_model_date) & 
                            (prev_forecast_df['ds'] <= current_date)
                        ]
                        
                        if not historical_period.empty:
                            fig_combined.add_trace(go.Scatter(
                                x=historical_period['ds'],
                                y=historical_period['yhat'],
                                line=dict(color='red', width=2, dash='dot'),
                                name='წინა პროგნოზი',
                                showlegend=True
                            ), row=1, col=1)
                            
                            # პროგნოზის ზუსტობის ანალიზი
                            accuracy_data = []
                            for _, row in historical_period.iterrows():
                                date = row['ds']
                                predicted = row['yhat']
                                
                                # რეალური ფასის პოვნა იმავე თარიღისთვის
                                if date in data_candle.index:
                                    actual = data_candle.loc[date, 'Close']
                                    error_pct = abs((predicted - actual) / actual) * 100
                                    accuracy_data.append({
                                        'date': date,
                                        'error_pct': error_pct,
                                        'predicted': predicted,
                                        'actual': actual
                                    })
                            
                            if accuracy_data:
                                accuracy_df = pd.DataFrame(accuracy_data)
                                
                                # ზუსტობის გრაფიკი
                                fig_combined.add_trace(go.Scatter(
                                    x=accuracy_df['date'],
                                    y=accuracy_df['error_pct'],
                                    line=dict(color='orange', width=2),
                                    name='შეცდომის %',
                                    showlegend=True
                                ), row=5, col=1)
                                
                                avg_error = accuracy_df['error_pct'].mean()
                                st.info(f"🎯 წინა პროგნოზის საშუალო შეცდომა: {avg_error:.2f}%")
                    
                    except Exception as e:
                        st.warning(f"წინა პროგნოზის ანალიზის შეცდომა: {e}")

                # ტექნიკური ინდიკატორების დამატება
                # მოცულობა
                colors = ['red' if close < open else 'green' for close, open in zip(data_candle['Close'], data_candle['Open'])]
                fig_combined.add_trace(go.Bar(
                    x=data_candle.index, 
                    y=data_candle['Volume'], 
                    name='Volume',
                    marker=dict(color=colors),
                    showlegend=False
                ), row=2, col=1)

                # RSI
                if 'RSI' in data_candle.columns:
                    fig_combined.add_trace(go.Scatter(
                        x=data_candle.index, 
                        y=data_candle['RSI'],
                        line=dict(color='blue', width=2), 
                        name='RSI',
                        showlegend=False
                    ), row=3, col=1)
                    
                    # RSI ზღვრები
                    fig_combined.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                    fig_combined.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

                # MACD
                if 'MACD' in data_candle.columns:
                    fig_combined.add_trace(go.Scatter(
                        x=data_candle.index, 
                        y=data_candle['MACD'],
                        line=dict(color='red', width=1), 
                        name='MACD',
                        showlegend=False
                    ), row=4, col=1)
                    
                    if 'MACD_Signal' in data_candle.columns:
                        fig_combined.add_trace(go.Scatter(
                            x=data_candle.index, 
                            y=data_candle['MACD_Signal'],
                            line=dict(color='blue', width=1), 
                            name='Signal',
                            showlegend=False
                        ), row=4, col=1)

                # გრაფიკის კონფიგურაცია
                fig_combined.update_layout(
                    height=1200, 
                    title_text=f"🎯 {ticker} სრული ანალიზი - ისტორია და პროგნოზი",
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark"
                )
                
                fig_combined.update_yaxes(title_text="ფასი ($)", row=1, col=1)
                fig_combined.update_yaxes(title_text="მოცულობა", row=2, col=1)
                fig_combined.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
                fig_combined.update_yaxes(title_text="MACD", row=4, col=1)
                fig_combined.update_yaxes(title_text="შეცდომა %", row=5, col=1)

                st.plotly_chart(fig_combined, use_container_width=True)

                # პროგნოზის სტატისტიკა
                current_price = data_candle['Close'].iloc[-1]
                future_forecast = forecast.tail(forecast_days)
                predicted_price = future_forecast['yhat'].iloc[-1]
                price_change_forecast = ((predicted_price - current_price) / current_price) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📈 მიმდინარე ფასი", f"${current_price:.4f}")
                
                with col2:
                    st.metric(f"📊 {forecast_days} დღის შემდეგ", f"${predicted_price:.4f}")
                
                with col3:
                    st.metric("📈 პროგნოზირებული ცვლილება", f"{price_change_forecast:+.2f}%")
                
                with col4:
                    trend = "📈 ზრდადი" if price_change_forecast > 0 else "📉 კლებადი"
                    st.metric("📊 ტრენდი", trend)

            else:
                st.warning("არასაკმარისი მონაცემები პროგნოზისთვის (საჭიროა მინიმუმ 30 დღე)")
                
        except Exception as e:
            st.error(f"პროგნოზის შეცდომა: {e}")
            import traceback
            st.text(traceback.format_exc())
    
    else:
        # ჩვეულებრივი ტექნიკური ანალიზი არა-დღიური ინტერვალებისთვის
        st.subheader(f"📊 {ticker} - ისტორიული ფასი და ტექნიკური ანალიზი")
        
        # --- Plotly გრაფიკების შექმნა ---
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=('ფასი და EMA', 'მოცულობა', 'RSI', 'MACD')
        )

        # სანთლის გრაფიკი
        fig.add_trace(go.Candlestick(
            x=data_candle.index,
            open=data_candle['Open'], 
            high=data_candle['High'],
            low=data_candle['Low'], 
            close=data_candle['Close'],
            name="Candlestick",
            showlegend=False
        ), row=1, col=1)

        # EMA ხაზების დამატება
        if 'EMA_9' in data_candle.columns:
            fig.add_trace(go.Scatter(
                x=data_candle.index, 
                y=data_candle['EMA_9'],
                line=dict(color='orange', width=1), 
                name='EMA 9'
            ), row=1, col=1)
        
        if 'EMA_21' in data_candle.columns:
            fig.add_trace(go.Scatter(
                x=data_candle.index, 
                y=data_candle['EMA_21'],
                line=dict(color='purple', width=1), 
                name='EMA 21'
            ), row=1, col=1)

        if 'EMA_50' in data_candle.columns:
            fig.add_trace(go.Scatter(
                x=data_candle.index, 
                y=data_candle['EMA_50'],
                line=dict(color='blue', width=1), 
                name='EMA 50'
            ), row=1, col=1)

        # Bollinger Bands-ის დამატება
        if 'BB_Upper' in data_candle.columns and 'BB_Lower' in data_candle.columns:
            fig.add_trace(go.Scatter(
                x=data_candle.index, 
                y=data_candle['BB_Upper'],
                line=dict(color='gray', width=1, dash='dash'), 
                name='BB Upper',
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data_candle.index, 
                y=data_candle['BB_Lower'],
                line=dict(color='gray', width=1, dash='dash'), 
                name='BB Lower',
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ), row=1, col=1)
        
        # მოცულობა
        colors = ['red' if close < open else 'green' for close, open in zip(data_candle['Close'], data_candle['Open'])]
        fig.add_trace(go.Bar(
            x=data_candle.index, 
            y=data_candle['Volume'], 
            name='Volume',
            marker=dict(color=colors),
            showlegend=False
        ), row=2, col=1)

        # RSI
        if 'RSI' in data_candle.columns:
            fig.add_trace(go.Scatter(
                x=data_candle.index, 
                y=data_candle['RSI'],
                line=dict(color='blue', width=2), 
                name='RSI',
                showlegend=False
            ), row=3, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=50, line_dash="solid", line_color="gray", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # MACD
        if 'MACD' in data_candle.columns:
            fig.add_trace(go.Scatter(
                x=data_candle.index, 
                y=data_candle['MACD'],
                line=dict(color='red', width=1), 
                name='MACD',
                showlegend=False
            ), row=4, col=1)
            
            if 'MACD_Signal' in data_candle.columns:
                fig.add_trace(go.Scatter(
                    x=data_candle.index, 
                    y=data_candle['MACD_Signal'],
                    line=dict(color='blue', width=1), 
                    name='Signal',
                    showlegend=False
                ), row=4, col=1)
            
            # MACD Histogram
            if 'MACD_Histogram' in data_candle.columns:
                colors = ['green' if val >= 0 else 'red' for val in data_candle['MACD_Histogram']]
                fig.add_trace(go.Bar(
                    x=data_candle.index, 
                    y=data_candle['MACD_Histogram'],
                    name='Histogram',
                    marker=dict(color=colors),
                    showlegend=False
                ), row=4, col=1)

        # გრაფიკის განლაგების კონფიგურაცია
        fig.update_layout(
            height=900, 
            title_text=f"🎯 {ticker} ტექნიკური ანალიზი",
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )
        
        fig.update_yaxes(title_text="ფასი ($)", row=1, col=1)
        fig.update_yaxes(title_text="მოცულობა", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # სტატისტიკა არა-დღიური ინტერვალებისთვის
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = data_candle['Close'].iloc[-1]
            st.metric("📈 მიმდინარე ფასი", f"${current_price:.4f}")
        
        with col2:
            price_change = current_price - data_candle['Close'].iloc[-2] if len(data_candle) > 1 else 0
            price_change_pct = (price_change / data_candle['Close'].iloc[-2] * 100) if len(data_candle) > 1 and data_candle['Close'].iloc[-2] != 0 else 0
            st.metric("📊 ცვლილება", f"${price_change:.4f}", f"{price_change_pct:.2f}%")
        
        with col3:
            if 'RSI' in data_candle.columns and not data_candle['RSI'].isnull().iloc[-1]:
                current_rsi = data_candle['RSI'].iloc[-1]
                st.metric("🎯 RSI", f"{current_rsi:.2f}")
            else:
                st.metric("🎯 RSI", "N/A")
        
        with col4:
            volume_24h = data_candle['Volume'].iloc[-1]
            st.metric("💰 მოცულობა", f"{volume_24h:,.0f}")

        st.warning("⚠️ პროგნოზი მხოლოდ დღიური (1d) ინტერვალისთვის ხელმისაწვდომია.")
        st.info("გთხოვთ აირჩიოთ '1d' ინტერვალი პროგნოზის სანახავად.")

    # --- სიგნალების გენერირების ნაწილი ---
    st.subheader(f"🚨 ტრეიდინგის სიგნალები")
    
    # მხოლოდ უახლესი მონაცემები
    recent_data = data_candle.tail(50).copy()
    
    # სიგნალების ანალიზი
    signals_list = []
    
    # მიმდინარე RSI სიგნალი
    if 'RSI' in recent_data.columns and not recent_data['RSI'].isnull().iloc[-1]:
        current_rsi = recent_data['RSI'].iloc[-1]
        if current_rsi > 70:
            signals_list.append({"სიგნალი": "🔴 გაყიდვა", "მიზეზი": "RSI Overbought (>70)", "ძალა": "ძლიერი"})
        elif current_rsi < 30:
            signals_list.append({"სიგნალი": "🟢 ყიდვა", "მიზეზი": "RSI Oversold (<30)", "ძალა": "ძლიერი"})
    
    # EMA Crossover სიგნალი
    if len(recent_data) >= 2 and 'EMA_9' in recent_data.columns and 'EMA_21' in recent_data.columns:
        ema9_current = recent_data['EMA_9'].iloc[-1]
        ema21_current = recent_data['EMA_21'].iloc[-1]
        ema9_prev = recent_data['EMA_9'].iloc[-2]
        ema21_prev = recent_data['EMA_21'].iloc[-2]
        
        if not any([pd.isna(x) for x in [ema9_current, ema21_current, ema9_prev, ema21_prev]]):
            if ema9_prev < ema21_prev and ema9_current > ema21_current:
                signals_list.append({"სიგნალი": "🟢 ყიდვა", "მიზეზი": "EMA 9/21 Golden Cross", "ძალა": "საშუალო"})
            elif ema9_prev > ema21_prev and ema9_current < ema21_current:
                signals_list.append({"სიგნალი": "🔴 გაყიდვა", "მიზეზი": "EMA 9/21 Death Cross", "ძალა": "საშუალო"})
    
    # MACD სიგნალი
    if ('MACD' in recent_data.columns and 'MACD_Signal' in recent_data.columns and len(recent_data) >= 2):
        macd_current = recent_data['MACD'].iloc[-1]
        signal_current = recent_data['MACD_Signal'].iloc[-1]
        macd_prev = recent_data['MACD'].iloc[-2]
        signal_prev = recent_data['MACD_Signal'].iloc[-2]
        
        if not any([pd.isna(x) for x in [macd_current, signal_current, macd_prev, signal_prev]]):
            if macd_prev < signal_prev and macd_current > signal_current:
                signals_list.append({"სიგნალი": "🟢 ყიდვა", "მიზეზი": "MACD Bullish Crossover", "ძალა": "საშუალო"})
            elif macd_prev > signal_prev and macd_current < signal_current:
                signals_list.append({"სიგნალი": "🔴 გაყიდვა", "მიზეზი": "MACD Bearish Crossover", "ძალა": "საშუალო"})
    
    # ფასის ტრენდის ანალიზი
    current_price = data_candle['Close'].iloc[-1]
    if len(data_candle) > 1:
        price_change = current_price - data_candle['Close'].iloc[-2]
        price_change_pct = (price_change / data_candle['Close'].iloc[-2] * 100) if data_candle['Close'].iloc[-2] != 0 else 0
        
        if abs(price_change_pct) > 5:
            if price_change_pct > 5:
                signals_list.append({"სიგნალი": "⚡ ძლიერი ზრდა", "მიზეზი": f"ფასი გაიზარდა {price_change_pct:.2f}%-ით", "ძალა": "ძლიერი"})
            else:
                signals_list.append({"სიგნალი": "⚡ ძლიერი ვარდნა", "მიზეზი": f"ფასი დაეცა {abs(price_change_pct):.2f}%-ით", "ძალა": "ძლიერი"})
    
    # სიგნალების ჩვენება
    if signals_list:
        signals_df = pd.DataFrame(signals_list)
        st.dataframe(signals_df, use_container_width=True)
    else:
        st.info("🔄 ნეიტრალური ბაზარი - მნიშვნელოვანი სიგნალები არ გამოვლენილა")
    
    # დამატებითი რეკომენდაციები
    st.subheader("💡 რეკომენდაციები")
    
    risk_level = "დაბალი"
    if 'RSI' in recent_data.columns and not recent_data['RSI'].isnull().iloc[-1]:
        current_rsi = recent_data['RSI'].iloc[-1]
        if current_rsi > 80 or current_rsi < 20:
            risk_level = "მაღალი"
        elif current_rsi > 70 or current_rsi < 30:
            risk_level = "საშუალო"
    else:
        current_rsi = 50  # default value
    
    # პროგნოზირებული ცვლილება (მხოლოდ დღიური ინტერვალისთვის)
    if interval == '1d' and 'price_change_forecast' in locals():
        forecast_info = f"- პროგნოზირებული ცვლილება: {price_change_forecast:+.2f}%"
        trend_info = f"- ტრენდი: {'ზრდადი' if price_change_forecast > 0 else 'კლებადი'}"
    else:
        forecast_info = "- პროგნოზი: მხოლოდ დღიური ინტერვალისთვის"
        trend_info = ""
    
    st.info(f"""
    **რისკის დონე:** {risk_level}
    
    **ზოგადი რეკომენდაცია:**
    - RSI: {current_rsi:.1f} ({'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'})
    {trend_info}
    {forecast_info}
    
    ⚠️ **გაფრთხილება:** ეს პროგნოზი მხოლოდ საინფორმაციო მიზნებისთვისაა და არ წარმოადგენს საინვესტიციო რჩევას.
    """)

    # წინა პროგნოზის ზუსტობის შეფასება (თუ ხელმისაწვდომია)
    if interval == '1d' and previous_forecast:
        st.subheader("📈 წინა პროგნოზის ანალიზი")
        
        try:
            prev_forecast_data = previous_forecast['forecast']
            prev_model_date = previous_forecast['model_date']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📅 წინა მოდელის თარიღი", prev_model_date.strftime("%Y-%m-%d"))
            
            with col2:
                if 'avg_error' in locals():
                    st.metric("🎯 საშუალო ზუსტობა", f"{100-avg_error:.1f}%")
                else:
                    st.metric("🎯 საშუალო ზუსტობა", "გაითვლება...")
            
            st.info("🔍 წითელი წერტილოვანი ხაზი მთავარ გრაფიკზე აჩვენებს წინა პროგნოზს რეალურ მონაცემებთან შესადარებლად")
            
        except Exception as e:
            st.warning(f"წინა პროგნოზის ანალიზის შეცდომა: {e}")

else:
    st.warning("⚠️ მონაცემების ჩატვირთვა ვერ მოხერხდა. გთხოვთ, სცადოთ სხვა კრიპტოვალუტა ან ინტერვალი.")
