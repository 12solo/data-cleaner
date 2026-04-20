import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Accurate Data Cleaner", layout="wide")
st.title("Stress-Strain Curve Anomaly Cleaner")

st.markdown("Upload your raw data. This app uses a **Robust Modified Z-Score (MAD)** algorithm to accurately pinpoint and remove equipment drops without affecting the true curve.")

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
    
    # A threshold of 3.5 means "flag any point that is 3.5x further from the trend than the normal noise"
    z_threshold = st.sidebar.slider("Anomaly Threshold (Modified Z-Score)", min_value=1.0, max_value=10.0, value=3.5, step=0.5)
    window_size = st.sidebar.slider("Rolling Window Size", min_value=3, max_value=15, value=5, step=2)

    # 1. Calculate the rolling median (the robust local trend)
    df['Rolling_Median'] = df[col_stress].rolling(window=window_size, center=True).median()
    df['Rolling_Median'] = df['Rolling_Median'].fillna(df[col_stress])
    
    # 2. Calculate the rolling Median Absolute Deviation (MAD)
    # This determines the normal "thickness" or "noise" of the line locally
    def calculate_mad(x):
        return np.median(np.abs(x - np.median(x)))
        
    df['Rolling_MAD'] = df[col_stress].rolling(window=window_size, center=True).apply(calculate_mad)
    
    # Prevent division by zero in perfectly flat areas by replacing 0 with a tiny number
    df['Rolling_MAD'] = df['Rolling_MAD'].replace(0, 1e-6)
    
    # 3. Calculate Modified Z-Score
    # Formula: 0.6745 * (value - median) / MAD
    df['Mod_Z_Score'] = 0.6745 * np.abs(df[col_stress] - df['Rolling_Median']) / df['Rolling_MAD']
    
    # 4. Identify the outliers
    outliers = df[df['Mod_Z_Score'] > z_threshold]
    df_cleaned = df[df['Mod_Z_Score'] <= z_threshold].copy()

    # --- Visualization ---
    st.subheader("Data Preview: Outlier Detection")
    
    fig = go.Figure()
    
    # Plot the original line
    fig.add_trace(go.Scatter(
        x=df[col_deform], y=df[col_stress], 
        mode='lines', 
        name='Original Data',
        line=dict(color='blue')
    ))
    
    # Highlight the detected outliers in RED
    fig.add_trace(go.Scatter(
        x=outliers[col_deform], y=outliers[col_stress], 
        mode='markers', 
        name='Detected Outliers',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(title="Red 'X' marks the data points scheduled for removal",
                      xaxis_title=col_deform,
                      yaxis_title=col_stress)
    
    st.plotly_chart(fig, use_container_width=True)

    # Clean up the dataframe for export
    df_cleaned = df_cleaned.drop(columns=['Rolling_Median', 'Rolling_MAD', 'Mod_Z_Score'])

    st.sidebar.success(f"Algorithm accurately pinpointed {len(outliers)} outlier(s).")

    # Download mechanism
    st.sidebar.markdown("---")
    csv = df_cleaned.to_csv(sep='\t', index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Cleaned Data",
        data=csv,
        file_name="accurately_corrected_data.txt",
        mime="text/plain",
    )
