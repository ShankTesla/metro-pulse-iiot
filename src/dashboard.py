import time
import json
import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as rt
from kafka import KafkaConsumer
import plotly.graph_objects as go

# --- CONFIGURATION ---
KAFKA_TOPIC = "metro-sensor-data"
BOOTSTRAP_SERVERS = ['localhost:19092']
MODEL_PATH = "models/metro_model.onnx"

# --- PAGE CONFIG ---
st.set_page_config(page_title="MetroPulse Monitor", page_icon="🚄", layout="wide")

# --- CUSTOM CSS FOR SCROLLABLE LOGS ---
st.markdown("""
    <style>
    .log-box {
        height: 200px;
        overflow-y: scroll;
        background-color: #0e1117; /* Force Black Background */
        color: #00FF00;            /* Force Bright Green Text */
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚄 MetroPulse: Edge Predictive Maintenance")

# --- INIT RESOURCES ---
@st.cache_resource
def load_model():
    return rt.InferenceSession(MODEL_PATH)

@st.cache_resource
def load_consumer():
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

try:
    sess = load_model()
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    consumer = load_consumer()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- LAYOUT ---
# Top: Chart
chart_col = st.empty()

# Bottom: Metrics (Left) and Logs (Right)
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("System Health")
    status_indicator = st.empty()
    metric_temp = st.empty()
    metric_current = st.empty()

with col2:
    st.subheader("Live Event Logs")
    log_placeholder = st.empty()

# --- STATE MANAGEMENT ---
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = []
if 'log_buffer' not in st.session_state:
    st.session_state.log_buffer = []

# --- STREAMING LOOP ---
MAX_POINTS = 100

for message in consumer:
    row = message.value
    timestamp = row['timestamp']
    
    # Extract features (15 total)
    features = [
        row['TP2'], row['TP3'], row['H1'], row['DV_pressure'],
        row['Reservoirs'], row['Oil_temperature'], row['Motor_current'],
        row['COMP'], row['DV_eletric'], row['Towers'], row['MPG'],
        row['LPS'], row['Pressure_switch'], row['Oil_level'], 
        row['Caudal_impulses']
    ]
    
    # Inference
    input_data = np.array([features], dtype=np.float32)
    pred_onx = sess.run([label_name], {input_name: input_data})[0]
    prediction = pred_onx[0] # 0 = Normal, 1 = Fail
    
    # 1. LOGIC: Update Buffers
    st.session_state.data_buffer.append({
        'time': timestamp, 
        'temp': row['Oil_temperature'],
        'status': 'red' if prediction == 1 else '#00CC96' # Color logic
    })
    if len(st.session_state.data_buffer) > MAX_POINTS:
        st.session_state.data_buffer.pop(0)

    # Log formatting
    status_text = "🔴 FAILURE" if prediction == 1 else "🟢 NORMAL"
    log_msg = f"[{timestamp}] {status_text} | Oil Temp: {row['Oil_temperature']:.2f}"
    st.session_state.log_buffer.insert(0, log_msg) # Add to top
    if len(st.session_state.log_buffer) > 50: # Keep only last 50 logs
        st.session_state.log_buffer.pop()

    # 2. VISUALS: Chart
    df_chart = pd.DataFrame(st.session_state.data_buffer)
    
    fig = go.Figure()
    # We draw lines. To make the line change color dynamically, 
    # we actually draw markers colored by status, connected by lines.
    fig.add_trace(go.Scatter(
        x=df_chart['time'], 
        y=df_chart['temp'], 
        mode='lines+markers',
        line=dict(color='gray', width=1),
        marker=dict(color=df_chart['status'], size=6),
        name='Oil Temp'
    ))
    
    fig.update_layout(
        title="Real-time Oil Temperature Sensor",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_range=[0, 100] # Fix y-axis so it doesn't jump around
    )
    chart_col.plotly_chart(fig, use_container_width=True)

    # 3. VISUALS: Metrics
    with col1:
        metric_temp.metric("Oil Temperature", f"{row['Oil_temperature']:.1f} °C")
        metric_current.metric("Motor Current", f"{row['Motor_current']:.2f} A")
        
        if prediction == 1:
            status_indicator.error("CRITICAL FAILURE DETECTED")
        else:
            status_indicator.success("System Normal")

    # 4. VISUALS: Custom Scrollable Logs
    log_html = "<br>".join(st.session_state.log_buffer)
    log_placeholder.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    # time.sleep(0.05) # Optional: Control speed