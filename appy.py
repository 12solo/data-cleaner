import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Tensile Data Cleaner", layout="wide")
st.title("Stress-Strain Curve Anomaly Cleaner")

st.markdown("Upload your raw data. This app uses a **Robust Modified Z-Score (MAD)** algorithm to pinpoint and remove equipment drops without affecting the true curve.")

# File uploader
uploaded_file = st.file_uploader("Upload Raw Data (TXT/CSV)", type=['txt', 'csv'])

if uploaded_file:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file, sep=r'\s+') 
    
    # Identify columns based on the data structure
    col_load = df.columns[0]
    col_deform = df.columns[1]
    col_stress = df.columns[2]

    # --- Automated Statistical Filtering ---
    st.sidebar.header("Filter Sensitivity")
    st.sidebar.markdown("The default threshold of 3.5 is the statistical standard for anomaly detection.")
    
    z_threshold = st.sidebar.slider("Anomaly Threshold (Modified Z-Score)", min_value=1.0, max_value=10.0, value=3.5, step=0.5)
    window_size = st.sidebar.slider("Rolling Window Size", min_value=3, max_value=15, value=5, step=2)

    # 1. Calculate the rolling median (the robust local trend)
    df['Rolling_Median'] = df[col_stress].rolling(window=window_size, center=True).median()
    df['Rolling_Median'] = df['Rolling_Median'].fillna(df[col_stress])
    
    # 2. Calculate the rolling Median Absolute Deviation (MAD)
    def calculate_mad(x):
        return np.median(np.abs(x - np.median(x)))
        
    df['Rolling_MAD'] = df[col_stress].rolling(window=window_size, center=True).apply(calculate_mad)
    df['Rolling_MAD'] = df['Rolling_MAD'].replace(0, 1e-6)
    
    # 3. Calculate Modified Z-Score
    df['Mod_Z_Score'] = 0.6745 * np.abs(df[col_stress] - df['Rolling_Median']) / df['Rolling_MAD']
    
    # 4. Identify the outliers and create the cleaned dataset
    outliers = df[df['Mod_Z_Score'] > z_threshold]
    df_cleaned = df[df['Mod_Z_Score'] <= z_threshold].copy()

    # --- Visualization 1: Outlier Detection ---
    st.subheader("1. Outlier Detection")
    
    fig_detect = go.Figure()
    
    # Plot the original line in deep blue
    fig_detect.add_trace(go.Scatter(
        x=df[col_deform], y=df[col_stress], 
        mode='lines', 
        name='Original Data',
        line=dict(color='#003366', width=2)
    ))
    
    # Highlight the detected outliers in RED
    fig_detect.add_trace(go.Scatter(
        x=outliers[col_deform], y=outliers[col_stress], 
        mode='markers', 
        name='Detected Outliers',
        marker=dict(color='#D32F2F', size=10, symbol='x', line=dict(width=2, color='white'))
    ))
    
    fig_detect.update_layout(
        title="Red 'X' marks the data points scheduled for removal",
        xaxis_title=col_deform,
        yaxis_title=col_stress,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333333'),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_detect.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    fig_detect.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    
    st.plotly_chart(fig_detect, use_container_width=True)


    # --- Visualization 2: Cleaned Data Preview ---
    st.subheader("2. Cleaned Data Preview")
    
    fig_clean = go.Figure()
    
    # Plot the cleaned line in deep blue
    fig_clean.add_trace(go.Scatter(
        x=df_cleaned[col_deform], y=df_cleaned[col_stress], 
        mode='lines', 
        name='Cleaned Data',
        line=dict(color='#003366', width=2)
    ))
    
    fig_clean.update_layout(
        title="Final Stress-Strain Curve (Anomalies Removed)",
        xaxis_title=col_deform,
        yaxis_title=col_stress,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333333'),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_clean.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    fig_clean.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    
    st.plotly_chart(fig_clean, use_container_width=True)

    # --- Export ---
    # Clean up the dataframe for export
    df_cleaned = df_cleaned.drop(columns=['Rolling_Median', 'Rolling_MAD', 'Mod_Z_Score'])

    st.sidebar.success(f"Algorithm accurately pinpointed and removed {len(outliers)} outlier(s).")

    st.sidebar.markdown("---")
    csv = df_cleaned.to_csv(sep='\t', index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Cleaned Data",
        data=csv,
        file_name="accurately_corrected_data.txt",
        mime="text/plain",
    )
