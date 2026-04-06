import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# 1. CONFIGURACIÓN Y ESTILO (OCULTAR DEPLOY)
st.set_page_config(page_title="Gold AI Monitor Pro", layout="wide")

st.markdown("""
    <style>
    .stAppDeployButton {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Autorefresh cada 5 minutos
st_autorefresh(interval=300000, key="gold_ultimate_v6")

# 2. OBTENCIÓN DE DATOS E INYECCIÓN "LIVE"
@st.cache_data(ttl=200)
def fetch_realtime_data():
    gold_ticker = yf.Ticker("GC=F")
    fast = gold_ticker.fast_info
    live_price = fast['last_price']
    prev_close = fast['previous_close']
    
    df = gold_ticker.history(period="5y")
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df['Date'] = df['Date'].dt.tz_localize(None)
    df.columns = ['ds', 'y']
    
    # Inyectamos el precio en vivo como último dato
    df.iloc[-1, df.columns.get_loc('y')] = live_price
    
    # Tipo de cambio
    eurusd = yf.Ticker("EURUSD=X")
    rate = 1 / eurusd.fast_info['last_price']
    
    return live_price, prev_close, df, rate

current_usd, prev_close_usd, df_hist, usd_to_eur = fetch_realtime_data()

# 3. PREDICCIÓN CON IA
@st.cache_resource(ttl=300)
def get_ai_forecast(df, trigger_val):
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    return forecast

forecast_raw = get_ai_forecast(df_hist, current_usd)

# --- CÁLCULOS DE MÉTRICAS ---
last_predicted_usd = forecast_raw[forecast_raw['ds'] <= df_hist['ds'].max()].iloc[-1]['yhat']
diff_usd = current_usd - last_predicted_usd
diff_pct = (diff_usd / last_predicted_usd) * 100
fiabilidad_decimal = (1 - abs(diff_usd / current_usd))
fiabilidad_porcentaje = max(0, 100 * fiabilidad_decimal)

# Valor Ajustado Actual
ai_ajustado_usd = last_predicted_usd * fiabilidad_decimal

daily_diff_usd = current_usd - prev_close_usd
daily_diff_pct = (daily_diff_usd / prev_close_usd) * 100
std_dev_usd = df_hist['y'].tail(30).std()

# 4. INTERFAZ (UI)
st.title(f"Oro Canillejas | Tasación Premium 24/7")
st.write(f"Actualización automática: **{datetime.now().strftime('%H:%M:%S')}**")

moneda = st.sidebar.radio("Ver en:", ["EUR (€)", "USD ($)"])
conversion = usd_to_eur if "EUR" in moneda else 1.0
simbolo = "€" if "EUR" in moneda else "$"

# 5. --- PANEL DE MÉTRICAS (5 COLUMNAS) ---
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(label="🔴 Precio en Vivo", value=f"{simbolo}{current_usd * conversion:,.2f}",
              delta=f"{daily_diff_usd * conversion:,.2f} ({daily_diff_pct:.2f}%)")

with m2:
    st.metric(label="🤖 Estimado IA", value=f"{simbolo}{last_predicted_usd * conversion:,.2f}",
              delta=f"{diff_usd * conversion:,.2f} ({diff_pct:.2f}%)", delta_color="inverse")

with m3:
    st.metric(label="🎯 AI Ajustado", value=f"{simbolo}{ai_ajustado_usd * conversion:,.2f}",
              help="Estimado IA multiplicado por el factor de fiabilidad.")

with m4:
    st.metric(label="✅ Fiabilidad IA", value=f"{fiabilidad_porcentaje:.2f}%",
              delta=f"Desv. Est: {std_dev_usd * conversion:.2f}", delta_color="normal")

with m5:
    st.metric(label="💱 Cambio USD/EUR", value=f"{usd_to_eur:.4f}")

st.divider()

# 6. GRÁFICO Plotly (CON LÍNEA AJUSTADA)
forecast_future_only = forecast_raw[forecast_raw['ds'] >= df_hist['ds'].max()].copy()
# Creamos la columna ajustada en el dataframe futuro
forecast_future_only['yhat_adj'] = forecast_future_only['yhat'] * fiabilidad_decimal

fig = go.Figure()

# Línea Real
fig.add_trace(go.Scatter(x=df_hist['ds'].tail(60), y=df_hist['y'].tail(60) * conversion, 
                         name="Precio Real", line=dict(color='#1f77b4', width=2)))

# Línea Predicción IA Original
fig.add_trace(go.Scatter(x=forecast_future_only['ds'], y=forecast_future_only['yhat'] * conversion, 
                         name="Predicción IA", line=dict(color='gold', dash='dot', width=2)))

# LÍNEA IA AJUSTADA (SOLICITADA)
fig.add_trace(go.Scatter(x=forecast_future_only['ds'], y=forecast_future_only['yhat_adj'] * conversion, 
                         name="IA Ajustada (Fiabilidad)", line=dict(color='#00ff00', width=3)))

fig.update_layout(hovermode="x unified", template="plotly_white", height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# 7. TABLA DE PRONÓSTICO SEMANAL CON COLUMNA AJUSTADA
st.subheader(f"📅 Pronóstico Próximos 7 Días ({moneda})")

# Seleccionamos columnas incluyendo la nueva 'yhat_adj'
tabla = forecast_future_only[['ds', 'yhat', 'yhat_adj', 'yhat_lower', 'yhat_upper']].tail(7).copy()

# Aplicar conversión
for col in ['yhat', 'yhat_adj', 'yhat_lower', 'yhat_upper']:
    tabla[col] = tabla[col] * conversion

# Formato de Fecha Día-Mes-Año
tabla['ds'] = pd.to_datetime(tabla['ds']).dt.strftime('%d-%m-%Y')

# Renombrar columnas para la tabla final
tabla.columns = ['Fecha', 'Precio Estimado', 'Precio Ajustado', 'Mínimo (Soporte)', 'Máximo (Resistencia)']

# Mostrar tabla
st.dataframe(tabla.style.format({
    'Precio Estimado': f'{simbolo}{{:.2f}}',
    'Precio Ajustado': f'{simbolo}{{:.2f}}',
    'Mínimo (Soporte)': f'{simbolo}{{:.2f}}',
    'Máximo (Resistencia)': f'{simbolo}{{:.2f}}'
}), use_container_width=True)

st.caption("Los datos se obtienen de Finance. La IA utiliza el modelo Prophet para detectar estacionalidad diaria.")

st.info("💡 La **IA Ajustada** (línea verde) aplica el porcentaje de fiabilidad actual a toda la proyección futura para corregir desviaciones históricas.")

st.warning("⚠️ El mercado de divisas y el oro cambian constantemente. La conversión utiliza el último tipo de cambio detectado.")

# Obtener el año actual dinámicamente
current_year = datetime.now().year

# Crear el footer con estilo Markdown
st.markdown("---")  # Línea divisoria
st.markdown(
    f"""
    <div style="text-align: center;">
        <p style="margin-bottom: 0;">Design by <b>David L.</b></p>
        <p style="margin-top: 0;">Created for <b>Compro Oro y Plata Canillejas</b> @{current_year}</p>
        <p style="font-size: 0.8em; color: gray;">All rights reserved.</p>
    </div>
    """, 
    unsafe_allow_html=True
)
