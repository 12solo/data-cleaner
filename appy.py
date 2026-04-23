import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter

def clean_tensile_data(df, col_deform, col_stress, min_def, max_def, window_size=5, z_threshold=3.5, apply_smoothing=False):
    """
    Cleans anomalies using MAD and replaces them with interpolated values 
    to maintain the physical line trend.
    """
    df_calc = df.copy().sort_values(by=col_deform)
    
    # 1. Calculate Modified Z-Score for anomaly detection
    df_calc['Rolling_Median'] = df_calc[col_stress].rolling(window=window_size, center=True).median().fillna(df_calc[col_stress])
    
    def calculate_mad(x):
        return np.median(np.abs(x - np.median(x)))
        
    df_calc['Rolling_MAD'] = df_calc[col_stress].rolling(window=window_size, center=True).apply(calculate_mad).replace(0, 1e-6)
    df_calc['Mod_Z_Score'] = 0.6745 * np.abs(df_calc[col_stress] - df_calc['Rolling_Median']) / df_calc['Rolling_MAD']
    
    # 2. Logic: Identify points to be "Repaired"
    in_target_region = (df_calc[col_deform] >= min_def) & (df_calc[col_deform] <= max_def)
    is_anomaly = df_calc['Mod_Z_Score'] > z_threshold
    
    # 3. Precision Cleaning: Interpolate rather than Drop
    # We set anomalous stress values to NaN, then interpolate
    df_calc.loc[is_anomaly & in_target_region, col_stress] = np.nan
    
    # 'linear' interpolation maintains the trend between the last known good points
    df_calc[col_stress] = df_calc[col_stress].interpolate(method='linear')
    
    # 4. Optional: Savitzky-Golay Smoothing (Preserves peaks while removing noise)
    if apply_smoothing:
        # Window length must be odd and greater than polynomial order (3)
        savgol_win = window_size if window_size % 2 != 0 else window_size + 1
        if len(df_calc) > savgol_win:
            df_calc[col_stress] = savgol_filter(df_calc[col_stress], savgol_win, 3)

    outliers = df[is_anomaly & in_target_region].copy()
    
    return df_calc.drop(columns=['Rolling_Median', 'Rolling_MAD', 'Mod_Z_Score']), outliers

# --- Streamlit UI ---
st.set_page_config(page_title="Precision Tensile Cleaner", layout="wide")
st.title("🔬 Precision Stress-Strain Cleaner")

uploaded_file = st.file_uploader("Upload Raw Data (TXT/CSV)", type=['txt', 'csv'])

if uploaded_file:
    # Flexible separator handling
    df = pd.read_csv(uploaded_file, sep=None, engine='python') 
    
    col_deform = df.columns[1]
    col_stress = df.columns[2]

    # --- Sidebar Controls ---
    st.sidebar.header("1. Target Region")
    min_deform = st.sidebar.number_input("Min Deformation", value=float(df[col_deform].min()))
    max_deform = st.sidebar.number_input("Max Deformation", value=float(df[col_deform].max()))

    st.sidebar.header("2. Sensitivity")
    z_thresh = st.sidebar.slider("Anomaly Sensitivity (Lower = Stricter)", 0.5, 10.0, 3.5)
    win_size = st.sidebar.slider("Analysis Window", 3, 51, 11, step=2)
    
    st.sidebar.header("3. Post-Processing")
    smooth_on = st.sidebar.checkbox("Apply Savitzky-Golay Smoothing", value=True)

    # Process
    df_cleaned, outliers = clean_tensile_data(df, col_deform, col_stress, min_deform, max_deform, win_size, z_thresh, smooth_on)

    # --- Visualization ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Anomaly Detection")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=df[col_deform], y=df[col_stress], name='Raw Data', line=dict(color='lightgrey')))
        fig_raw.add_trace(go.Scatter(x=outliers[col_deform], y=outliers[col_stress], mode='markers', name='Detected Slips', marker=dict(color='red', symbol='x')))
        fig_raw.update_layout(template="plotly_white", margin=dict(l=0,r=0,b=0,t=30))
        st.plotly_chart(fig_raw, use_container_width=True)

    with col2:
        st.subheader("Cleaned Trend")
        fig_clean = go.Figure()
        fig_clean.add_trace(go.Scatter(x=df_cleaned[col_deform], y=df_cleaned[col_stress], name='Cleaned/Interpolated', line=dict(color='blue', width=2)))
        fig_clean.update_layout(template="plotly_white", margin=dict(l=0,r=0,b=0,t=30))
        st.plotly_chart(fig_clean, use_container_width=True)

    # --- Export ---
    st.success(f"Successfully repaired {len(outliers)} points. Trend preserved via linear interpolation.")
    
    csv = df_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button("Download Precision Cleaned Data", data=csv, file_name="cleaned_tensile_data.csv", mime="text/csv")
