import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(page_title="VN Portfolio & Monte Carlo Optimizer", layout="wide")
st.title("📈 VN Portfolio & Monte Carlo Optimizer")
st.write("Track your Vietnamese stock portfolio performance against the VN-Index and discover the optimal asset allocation.")

# --- Session State Initialization ---
# This ensures that the stock inputs persist as the user interacts with the app
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = {}

# --- Sidebar Configuration ---
st.sidebar.header("1. Define Timeframe")
# Defaulting to the dates provided in the Jupyter output
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2026-06-29'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))

st.sidebar.header("2. Build Portfolio")
col1, col2 = st.sidebar.columns([2, 1])
with col1:
    ticker_input = st.text_input("Ticker (e.g., FPT, MBB)").strip().upper().replace('.VN', '')
with col2:
    shares_input = st.number_input("Shares", min_value=1.0, value=100.0, step=10.0)

if st.sidebar.button("Add Stock"):
    if ticker_input:
        st.session_state['portfolio'][ticker_input] = shares_input
    else:
        st.sidebar.warning("Please enter a valid ticker.")

st.sidebar.subheader("Current Holdings")
if not st.session_state['portfolio']:
    st.sidebar.info("No stocks added yet.")
else:
    for t, s in st.session_state['portfolio'].items():
        st.sidebar.write(f"- **{t}**: {s:,.0f} shares")
    
    if st.sidebar.button("Clear All"):
        st.session_state['portfolio'] = {}
        st.rerun()

st.sidebar.markdown("---")
run_analysis = st.sidebar.button("Run Analysis", type="primary")

# --- Main Application Logic ---
if run_analysis:
    if not st.session_state['portfolio']:
        st.warning("Please add at least one stock to your portfolio in the sidebar before running the analysis.")
    else:
        with st.spinner("Fetching Data and Running Simulations..."):
            portfolio_input = st.session_state['portfolio']
            
            # 1. Fetch Data using DNSE public API (for stocks)
            def fetch_dnse_close(symbol, start, end):
                start_timestamp = int(pd.Timestamp(start, tz="UTC").timestamp())
                end_timestamp = int(pd.Timestamp(end, tz="UTC").timestamp())
                url = (
                    "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
                    f"?resolution=1D&symbol={symbol}&from={start_timestamp}&to={end_timestamp}"
                )
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                payload = response.json()

                if not payload.get("t") or not payload.get("c"):
                    return pd.Series(dtype=float, name=symbol)

                prices = pd.DataFrame({
                    "date": pd.to_datetime(payload["t"], unit="s", utc=True).dt.tz_convert("Asia/Ho_Chi_Minh").dt.normalize().dt.tz_localize(None),
                    symbol: pd.to_numeric(payload["c"], errors="coerce"),
                })
                return prices.dropna(subset=[symbol]).drop_duplicates("date").set_index("date")[symbol]

            try:
                stock_series = {ticker: fetch_dnse_close(ticker, start_date, end_date) for ticker in portfolio_input}
                stock_data = pd.concat(stock_series, axis=1).sort_index()
                stock_data = stock_data.loc[start_date:end_date].dropna(how="all")
                
                if stock_data.empty:
                    st.error("DNSE returned no stock data for the selected date range.")
                    st.stop()
            except Exception as e:
                st.error(f"Failed to fetch stock data from DNSE: {e}")
                st.stop()

            # 2. Fetch Data using DNSE public API (for VN-Index only)
            try:
                # Convert dates to Unix timestamps
                start_ts = int(pd.Timestamp(start_date).timestamp())
                end_ts = int(pd.Timestamp(end_date).timestamp())
                url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/index?resolution=1D&symbol=VNINDEX&from={start_ts}&to={end_ts}"
                
                response = requests.get(url).json()
                benchmark_data = pd.DataFrame({
                    'time': pd.to_datetime(response['t'], unit='s'),
                    'close': response['c']
                })
                benchmark_data.set_index('time', inplace=True)
                benchmark_data.index = benchmark_data.index.tz_localize('UTC').tz_convert('Asia/Ho_Chi_Minh').normalize().tz_localize(None)
                
                # Filter to exact date range
                benchmark_data = benchmark_data.loc[start_date:end_date]
                benchmark_close = benchmark_data['close']
            except Exception as e:
                st.error(f"Failed to fetch VNINDEX via API: {e}")
                benchmark_close = pd.Series(dtype=float)

            # 3. Calculate Current Portfolio Performance
            portfolio_value = pd.DataFrame(index=stock_data.index)
            portfolio_value['Total Value'] = 0

            for ticker, shares in portfolio_input.items():
                portfolio_value['Total Value'] += stock_data[ticker].fillna(method='ffill') * shares

            portfolio_value = portfolio_value.dropna(subset=['Total Value'])
            portfolio_normalized = (portfolio_value['Total Value'] / portfolio_value['Total Value'].iloc[0]) * 100
            
            if not benchmark_close.empty:
                benchmark_normalized = (benchmark_close / benchmark_close.iloc[0]) * 100

            # --- VISUALIZATION 1: Performance Benchmark ---
            st.subheader("Performance Benchmarking")
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                x=portfolio_normalized.index, y=portfolio_normalized.values,
                mode='lines', name='My Portfolio', line=dict(color='#2962FF', width=2.5),
                hovertemplate='%{x|%Y-%m-%d}<br>Portfolio Growth: %{y:.2f}<extra></extra>'
            ))

            if not benchmark_close.empty:
                fig_perf.add_trace(go.Scatter(
                    x=benchmark_normalized.index, y=benchmark_normalized.values,
                    mode='lines', name='VN-Index', line=dict(color='#FF6D00', width=2, dash='dash'),
                    hovertemplate='%{x|%Y-%m-%d}<br>VN-Index Growth: %{y:.2f}<extra></extra>'
                ))

            fig_perf.update_layout(
                title='<b>Portfolio Performance vs VN-Index</b> (Base 100)',
                xaxis_title='Date', yaxis_title='Normalized Growth',
                template='plotly_white', hovermode='x unified',
                legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_perf, use_container_width=True)

            # 4. Monte Carlo Portfolio Optimization
            returns = stock_data.pct_change().dropna()
            latest_prices = stock_data.iloc[-1]
            current_values = np.array([portfolio_input[t] * latest_prices[t] for t in stock_data.columns])
            current_weights = current_values / current_values.sum()

            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            risk_free_rate = 0.07  # As defined in the Jupyter notebook

            # Reduced iterations from 100,000 to 10,000 for web app performance
            num_portfolios = 10000 
            results = np.zeros((3, num_portfolios))
            all_weights = np.zeros((num_portfolios, len(stock_data.columns)))

            for i in range(num_portfolios):
                weights = np.random.random(len(stock_data.columns))
                weights /= np.sum(weights)
                all_weights[i,:] = weights
                
                port_return = np.sum(expected_returns * weights)
                port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                results[0,i] = port_std_dev          
                results[1,i] = port_return           
                results[2,i] = (port_return - risk_free_rate) / port_std_dev  

            max_sharpe_idx = np.argmax(results[2])
            optimal_weights = all_weights[max_sharpe_idx, :]
            curr_return = np.sum(expected_returns * current_weights)
            curr_volatility = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))

            # --- VISUALIZATION 2: Optimization Results ---
            st.markdown("---")
            st.subheader("Optimization Results (10,000 Iterations)")

            # Format the output table
            opt_df = pd.DataFrame({
                'Stock': stock_data.columns,
                'Current Weight (%)': (current_weights * 100).round(2),
                'Optimal Weight (%)': (optimal_weights * 100).round(2)
            })
            st.dataframe(opt_df, use_container_width=True)

            fig_opt = make_subplots(
                rows=1, cols=2, 
                subplot_titles=('<b>Efficient Frontier</b>', '<b>Holdings: Current vs Optimal</b>'),
                horizontal_spacing=0.1
            )

            fig_opt.add_trace(go.Scatter(
                x=results[0, :], y=results[1, :], mode='markers',
                marker=dict(color=results[2, :], colorscale='Viridis', size=4, opacity=0.4, colorbar=dict(title='Sharpe Ratio', x=0.45), showscale=True),
                name='Simulated Portfolios', hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>'
            ), row=1, col=1)

            fig_opt.add_trace(go.Scatter(
                x=[results[0, max_sharpe_idx]], y=[results[1, max_sharpe_idx]], mode='markers',
                marker=dict(symbol='star', size=16, color='red', line=dict(width=1, color='black')),
                name='Max Sharpe (Optimal)'
            ), row=1, col=1)

            fig_opt.add_trace(go.Scatter(
                x=[curr_volatility], y=[curr_return], mode='markers',
                marker=dict(symbol='x', size=12, color='black', line=dict(width=2, color='black')),
                name='Current Portfolio'
            ), row=1, col=1)

            fig_opt.add_trace(go.Bar(
                x=list(stock_data.columns), y=current_weights * 100, name='Current Weight (%)', marker_color='#42A5F5'
            ), row=1, col=2)

            fig_opt.add_trace(go.Bar(
                x=list(stock_data.columns), y=optimal_weights * 100, name='Optimal Weight (%)', marker_color='#26A69A'
            ), row=1, col=2)

            fig_opt.update_layout(template='plotly_white', barmode='group', margin=dict(l=40, r=40, t=60, b=40))
            fig_opt.update_xaxes(title_text='Annualized Volatility (Risk)', row=1, col=1)
            fig_opt.update_yaxes(title_text='Expected Annual Return', row=1, col=1)
            fig_opt.update_yaxes(title_text='Allocation (%)', row=1, col=2)

            st.plotly_chart(fig_opt, use_container_width=True)