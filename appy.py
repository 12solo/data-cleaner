import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import io

def clean_tensile_data(df, col_deform, col_stress, min_def, max_def, window_size=5, z_threshold=3.5, apply_smoothing=False):
    """
    Precision cleaner that uses MAD, dilates the anomaly mask to catch the 'walls' 
    of the slip, and interpolates the removed region.
    """
    # Sort and reset index to ensure clean sequential operations
    df_calc = df.copy().sort_values(by=col_deform).reset_index(drop=True)
    
    # 1. Calculate Rolling Median and MAD
    df_calc['Rolling_Median'] = df_calc[col_stress].rolling(window=window_size, center=True).median().fillna(df_calc[col_stress])
    
    def calculate_mad(x):
        return np.median(np.abs(x - np.median(x)))
        
    df_calc['Rolling_MAD'] = df_calc[col_stress].rolling(window=window_size, center=True).apply(calculate_mad).replace(0, 1e-6).fillna(1e-6)
    
    # 2. Calculate Modified Z-Score
    df_calc['Mod_Z_Score'] = 0.6745 * np.abs(df_calc[col_stress] - df_calc['Rolling_Median']) / df_calc['Rolling_MAD']
    
    # 3. Target boundaries and initial detection
    in_target_region = (df_calc[col_deform] >= min_def) & (df_calc[col_deform] <= max_def)
    initial_anomalies = df_calc['Mod_Z_Score'] > z_threshold
    
    # ACCURACY UPGRADE: Mask Dilation
    dilated_anomalies = initial_anomalies | initial_anomalies.shift(1).fillna(False) | initial_anomalies.shift(-1).fillna(False)
    
    # Apply to target region only
    final_anomaly_mask = dilated_anomalies & in_target_region
    
    # 4. Precision Cleaning: Set to NaN and Interpolate
    df_calc.loc[final_anomaly_mask, col_stress] = np.nan
    df_calc[col_stress] = df_calc[col_stress].interpolate(method='linear')
    
    # 5. Optional Smoothing
    if apply_smoothing:
        # Window length must be odd and greater than polynomial order (3)
        savgol_win = int(window_size) if int(window_size) % 2 != 0 else int(window_size) + 1
        if len(df_calc) > savgol_win:
            df_calc[col_stress] = savgol_filter(df_calc[col_stress], savgol_win, 3)

    outliers = df[final_anomaly_mask].copy()
    
    cols_to_drop = ['Rolling_Median', 'Rolling_MAD', 'Mod_Z_Score']
    return df_calc.drop(columns=cols_to_drop), outliers

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

    # --- Top Visualization: Raw Data ---
    st.markdown("---")
    st.subheader("Raw Data & Detected Anomalies")
    fig_raw = go.Figure()
    
    fig_raw.add_trace(go.Scatter(x=df[col_deform], y=df[col_stress], name='Raw Data', line=dict(color='lightgrey', width=2)))
    fig_raw.add_trace(go.Scatter(x=outliers[col_deform], y=outliers[col_stress], mode='markers', name='Detected Slips', marker=dict(color='red', size=8, symbol='x')))
    
    # Bounding box lines
    fig_raw.add_vline(x=min_deform, line_width=1, line_dash="dash", line_color="gray", annotation_text="Min Target")
    fig_raw.add_vline(x=max_deform, line_width=1, line_dash="dash", line_color="gray", annotation_text="Max Target")
    
    fig_raw.update_layout(
        template="plotly_white", 
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=col_deform,
        yaxis_title=col_stress
    )
    st.plotly_chart(fig_raw, use_container_width=True)

    # --- Bottom Visualization: Cleaned Data ---
    st.markdown("---")
    st.subheader("Cleaned Trend")
    fig_clean = go.Figure()
    
    fig_clean.add_trace(go.Scatter(x=df_cleaned[col_deform], y=df_cleaned[col_stress], name='Cleaned/Interpolated', line=dict(color='#003366', width=2)))
    
    fig_clean.update_layout(
        template="plotly_white", 
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=col_deform,
        yaxis_title=col_stress
    )
    st.plotly_chart(fig_clean, use_container_width=True)

    # --- Export ---
    st.markdown("---")
    st.success(f"Successfully removed and repaired **{len(outliers)}** anomalous points. Trend preserved via linear interpolation.")
    
    # Prepare files for download
    # 1. CSV
    csv_data = df_cleaned.to_csv(index=False).encode('utf-8')
    
    # 2. TXT (Tab separated)
    txt_data = df_cleaned.to_csv(sep='\t', index=False).encode('utf-8')
    
    # 3. Excel (XLSX) - requires io.BytesIO to write to memory
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned Data')
    excel_data = excel_buffer.getvalue()

    # Layout buttons side-by-side
    st.write("### Download Cleaned Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Download as CSV", 
            data=csv_data, 
            file_name="cleaned_tensile_data.csv", 
            mime="text/csv"
        )
        
    with col2:
        st.download_button(
            label="📝 Download as TXT", 
            data=txt_data, 
            file_name="cleaned_tensile_data.txt", 
            mime="text/plain"
        )
        
    with col3:
        st.download_button(
            label="📊 Download as Excel", 
            data=excel_data, 
            file_name="cleaned_tensile_data.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
