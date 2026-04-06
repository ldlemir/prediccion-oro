import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

st.markdown("""
    <style>
    .stAppDeployButton {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Oro Canillejas", page_icon="img/logo2.webp",layout="wide")
# Puedes ajustar el ancho con 'width'
# Autorefresh cada 5 minutos
st_autorefresh(interval=300000, key="gold_final_v2")

# 2. OBTENCIÓN DE DATOS E INYECCIÓN "LIVE"
@st.cache_data(ttl=200)
def fetch_realtime_data():
    gold_ticker = yf.Ticker("GC=F")
    fast = gold_ticker.fast_info
    live_price = fast['last_price']
    prev_close = fast['previous_close']
    
    # Historial para entrenamiento de la IA
    df = gold_ticker.history(period="5y")
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df['Date'] = df['Date'].dt.tz_localize(None)
    df.columns = ['ds', 'y']
    
    # INYECCIÓN: El precio en vivo se vuelve el último punto del historial
    df.iloc[-1, df.columns.get_loc('y')] = live_price
    
    # Divisa (USD/EUR)
    eurusd = yf.Ticker("EURUSD=X")
    rate = 1 / eurusd.fast_info['last_price']
    
    return live_price, prev_close, df, rate

current_usd, prev_close_usd, df_hist, usd_to_eur = fetch_realtime_data()

# 3. PREDICCIÓN CON INTELIGENCIA ARTIFICIAL
@st.cache_resource(ttl=300)
def get_ai_forecast(df, trigger_val):
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    return forecast

forecast_raw = get_ai_forecast(df_hist, current_usd)

# --- CÁLCULOS DE MÉTRICAS ---

# IA Metrics
last_predicted_usd = forecast_raw[forecast_raw['ds'] <= df_hist['ds'].max()].iloc[-1]['yhat']
diff_usd = current_usd - last_predicted_usd
diff_pct = (diff_usd / last_predicted_usd) * 100
fiabilidad = max(0, 100 * (1 - abs(diff_usd / current_usd)))

# Diferencia Real (Hoy vs Cierre anterior)
daily_diff_usd = current_usd - prev_close_usd
daily_diff_pct = (daily_diff_usd / prev_close_usd) * 100

# Volatilidad (Desviación Estándar 30 días)
std_dev_usd = df_hist['y'].tail(30).std()

# 4. INTERFAZ (UI)
#st.image("img/logo2.webp", width=400)
st.title(f"Oro Canillejas | Tasación Premium 24/7")
st.write(f"Actualización automática: **{datetime.now().strftime('%H:%M:%S')}**")
st.write(f"Hoy es: **{datetime.now().strftime('%d-%m-%Y')}** | Los datos se actualizan solos.")

# Selector de moneda
moneda = st.sidebar.radio("Ver en:", ["EUR (€)", "USD ($)"])
conversion = usd_to_eur if "EUR" in moneda else 1.0
simbolo = "€" if "EUR" in moneda else "$"

# 5. --- PANEL DE MÉTRICAS ACTUALIZADO ---
m1, m2, m3, m4 = st.columns(4)

with m1:
    # Precio Real con su variación diaria
    st.metric(
        label="🔴 Precio en Vivo", 
        value=f"{simbolo}{current_usd * conversion:,.2f}",
        delta=f"{daily_diff_usd * conversion:,.2f} ({daily_diff_pct:.2f}%)"
    )

with m2:
    # Estimado IA con Diferencia de Precio + Porcentaje (SOLICITADO)
    st.metric(
        label="🤖 Estimado IA (Hoy)", 
        value=f"{simbolo}{last_predicted_usd * conversion:,.2f}",
        delta=f"{diff_usd * conversion:,.2f} ({diff_pct:.2f}%)",
        delta_color="inverse"
    )

with m3:
    # Fiabilidad con Desviación Estándar debajo
    st.metric(
        label="🎯 Fiabilidad IA", 
        value=f"{fiabilidad:.2f}%",
        delta=f"Desv. Est: {std_dev_usd * conversion:.2f}",
        delta_color="normal"
    )

with m4:
    # Tipo de Cambio
    st.metric(
        label="💱 Cambio USD/EUR", 
        value=f"{usd_to_eur:.4f}"
    )

st.divider()

def print_footer():
    # Obtiene el año actual automáticamente
    current_year = datetime.now().year
    
    # Estructura del footer
    footer = (
        f"Design by ldlemir\n"
        f"Created for Compro Oro Canillejas @{current_year}\n"
        f"All rights reserved."
    )
    
    print("-" * 30)
    print(footer)
    print("-" * 30)

# Llamada a la función
print_footer()

# 6. GRÁFICO Plotly
fig = go.Figure()
# Histórico
fig.add_trace(go.Scatter(x=df_hist['ds'].tail(60), y=df_hist['y'].tail(60) * conversion, name="Precio Real", line=dict(color='#1f77b4')))
# Predicción
forecast_future_only = forecast_raw[forecast_raw['ds'] >= df_hist['ds'].max()]
fig.add_trace(go.Scatter(x=forecast_future_only['ds'], y=forecast_future_only['yhat'] * conversion, name="Proyección IA", line=dict(color='gold', dash='dot')))
fig.update_layout(hovermode="x unified", template="plotly_white", height=400)
st.plotly_chart(fig, use_container_width=True)

# 7. TABLA DE PRONÓSTICO SEMANAL
st.subheader(f"📅 Pronóstico Próximos 7 Días ({moneda})")

# Limpieza de las 4 columnas seleccionadas para evitar errores de Pandas
tabla = forecast_future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).copy()

for col in ['yhat', 'yhat_lower', 'yhat_upper']:
    tabla[col] = tabla[col] * conversion

# Formato Fecha y Títulos
tabla['ds'] = pd.to_datetime(tabla['ds']).dt.strftime('%d-%m-%Y')
tabla.columns = ['Fecha', 'Precio Estimado', 'Mínimo (Soporte)', 'Máximo (Resistencia)']

st.dataframe(tabla.style.format({
    'Precio Estimado': f'{simbolo}{{:.2f}}',
    'Mínimo (Soporte)': f'{simbolo}{{:.2f}}',
    'Máximo (Resistencia)': f'{simbolo}{{:.2f}}'
}), use_container_width=True)


st.caption("Los datos se obtienen de Finance. La IA utiliza el modelo Prophet para detectar estacionalidad diaria.")

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