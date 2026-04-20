import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def remove_outliers_targeted(df, col_deform, col_stress, min_def, max_def, window_size=5, z_threshold=3.5):
    """Removes single-point anomalies using MAD, but only within a user-specified deformation range."""
    df_calc = df.copy()
    
    # 1. Calculate rolling median
    df_calc['Rolling_Median'] = df_calc[col_stress].rolling(window=window_size, center=True).median()
    df_calc['Rolling_Median'] = df_calc['Rolling_Median'].fillna(df_calc[col_stress])
    
    # 2. Calculate rolling MAD
    def calculate_mad(x):
        return np.median(np.abs(x - np.median(x)))
        
    df_calc['Rolling_MAD'] = df_calc[col_stress].rolling(window=window_size, center=True).apply(calculate_mad)
    df_calc['Rolling_MAD'] = df_calc['Rolling_MAD'].replace(0, 1e-6)
    
    # 3. Calculate Modified Z-Score
    df_calc['Mod_Z_Score'] = 0.6745 * np.abs(df_calc[col_stress] - df_calc['Rolling_Median']) / df_calc['Rolling_MAD']
    
    # 4. Filter logic: Must be an anomaly AND fall within the target deformation region
    in_target_region = (df_calc[col_deform] >= min_def) & (df_calc[col_deform] <= max_def)
    is_anomaly = df_calc['Mod_Z_Score'] > z_threshold
    
    outliers = df_calc[is_anomaly & in_target_region].copy()
    df_cleaned = df_calc[~(is_anomaly & in_target_region)].copy()
    
    # 5. Clean up calculation columns
    cols_to_drop = ['Rolling_Median', 'Rolling_MAD', 'Mod_Z_Score']
    outliers = outliers.drop(columns=cols_to_drop)
    df_cleaned = df_cleaned.drop(columns=cols_to_drop)
    
    return df_cleaned, outliers

# --- Streamlit UI ---
st.set_page_config(page_title="Tensile Data Cleaner", layout="wide")
st.title("Stress-Strain Anomaly Cleaner")
st.markdown("Upload your raw data. Define the region of the equipment slip, and the algorithm will accurately remove the bad data points.")

# File uploader
uploaded_file = st.file_uploader("Upload Raw Data (TXT/CSV)", type=['txt', 'csv'])

if uploaded_file:
    # Read the data
    df = pd.read_csv(uploaded_file, sep=r'\s+') 
    
    # Identify columns based on your structure
    col_deform = df.columns[1]
    col_stress = df.columns[2]

    # --- Filtering Section ---
    st.sidebar.header("Filter Settings")
    st.sidebar.markdown("Define the region of the equipment error.")
    
    # Interactive sliders/inputs to target the specific drop
    # Default values are set around the known anomaly at ~60mm
    min_deform = st.sidebar.number_input("Target Range: Min Deformation (mm)", value=55.0)
    max_deform = st.sidebar.number_input("Target Range: Max Deformation (mm)", value=65.0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Algorithm Sensitivity")
    z_threshold = st.sidebar.slider("Anomaly Threshold", min_value=1.0, max_value=10.0, value=3.5, step=0.5)
    window_size = st.sidebar.slider("Rolling Window Size", min_value=3, max_value=15, value=5, step=2)

    # Process the data with the targeted boundaries
    df_cleaned, outliers = remove_outliers_targeted(df, col_deform, col_stress, min_deform, max_deform, window_size, z_threshold)

    # --- Top Visualization: Raw Data ---
    st.subheader("Raw Data & Detected Anomalies")
    fig_raw = go.Figure()
    
    # Deep blue line for raw data
    fig_raw.add_trace(go.Scatter(
        x=df[col_deform], y=df[col_stress], 
        mode='lines', 
        name='Raw Data',
        line=dict(color='#003366', width=2) 
    ))
    
    # Red markers for the anomalies
    fig_raw.add_trace(go.Scatter(
        x=outliers[col_deform], y=outliers[col_stress], 
        mode='markers', 
        name='Anomalies',
        marker=dict(color='#D32F2F', size=8, symbol='x')
    ))

    # Add vertical lines to show the targeted bounding box on the graph
    fig_raw.add_vline(x=min_deform, line_width=1, line_dash="dash", line_color="gray", annotation_text="Min Def")
    fig_raw.add_vline(x=max_deform, line_width=1, line_dash="dash", line_color="gray", annotation_text="Max Def")
    
    fig_raw.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(title=col_deform, showgrid=True, gridcolor='#C0C0C0'),
        yaxis=dict(title=col_stress, showgrid=True, gridcolor='#C0C0C0'),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_raw, use_container_width=True)

    st.markdown("---")

    # --- Bottom Visualization: Cleaned Data ---
    st.subheader("Cleaned Data")
    fig_clean = go.Figure()
    
    # Deep blue line for cleaned data
    fig_clean.add_trace(go.Scatter(
        x=df_cleaned[col_deform], y=df_cleaned[col_stress], 
        mode='lines', 
        name='Cleaned Data',
        line=dict(color='#003366', width=2)
    ))
    
    fig_clean.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(title=col_deform, showgrid=True, gridcolor='#C0C0C0'),
        yaxis=dict(title=col_stress, showgrid=True, gridcolor='#C0C0C0'),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_clean, use_container_width=True)

    # --- Output & Export ---
    st.success(f"Processing complete! The algorithm detected and removed **{len(outliers)}** outlier point(s) between {min_deform}mm and {max_deform}mm.")

    csv = df_cleaned.to_csv(sep='\t', index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data",
        data=csv,
        file_name="cleaned_tensile_data.txt",
        mime="text/plain",
    )
