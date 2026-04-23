import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import io

def clean_tensile_data(df, col_deform, col_stress, min_def, max_def, window_size=5, z_threshold=3.5, 
                       apply_smoothing=False, remove_breakpoint=True, anomaly_action='Delete Rows', dilation_pts=2):
    """
    Bulletproof cleaner that respects original row order and uses hard index-dropping.
    """
    df_calc = df.copy()
    
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
    
    # 4. DYNAMIC MASK DILATION: Erase the 'walls' around the slip
    if dilation_pts > 0:
        dilated_anomalies = initial_anomalies.rolling(window=(2 * dilation_pts) + 1, center=True, min_periods=1).max().astype(bool)
    else:
        dilated_anomalies = initial_anomalies

    final_anomaly_mask = dilated_anomalies & in_target_region
    
    # Save outliers for the graph BEFORE we delete them
    outliers = df_calc[final_anomaly_mask].copy()

    # 5. HARD DELETION
    if anomaly_action == 'Delete Rows':
        df_calc = df_calc.drop(index=outliers.index)
    else:
        df_calc.loc[final_anomaly_mask, col_stress] = np.nan
        df_calc[col_stress] = df_calc[col_stress].interpolate(method='linear', limit_direction='both')
    
    # 6. Breakpoint / Fracture Removal
    if remove_breakpoint:
        peak_idx = df_calc[col_stress].idxmax()
        post_peak = df_calc.loc[peak_idx:].copy()
        
        if len(post_peak) > 1:
            steepest_drop_idx = post_peak[col_stress].diff().idxmin()
            if pd.notna(steepest_drop_idx):
                bad_tail_indexes = df_calc.loc[steepest_drop_idx:].index
                df_calc = df_calc.drop(index=bad_tail_indexes)

    # 7. Optional Smoothing
    if apply_smoothing:
        savgol_win = int(window_size) if int(window_size) % 2 != 0 else int(window_size) + 1
        if len(df_calc) > savgol_win:
            df_calc[col_stress] = savgol_filter(df_calc[col_stress], savgol_win, 3)

    # 8. Final Cleanup
    cols_to_drop = ['Rolling_Median', 'Rolling_MAD', 'Mod_Z_Score']
    cols_to_drop = [c for c in cols_to_drop if c in df_calc.columns]
    
    df_final = df_calc.drop(columns=cols_to_drop).dropna(subset=[col_deform, col_stress]).reset_index(drop=True)
    return df_final, outliers

# --- Streamlit UI ---
st.set_page_config(page_title="Precision Tensile Cleaner", layout="wide")
st.title("🔬 Precision Stress-Strain Cleaner")

uploaded_file = st.file_uploader("Upload Raw Data (TXT/CSV)", type=['txt', 'csv'])

if uploaded_file:
    # Try tab-separated first, fallback to whitespace if needed
    try:
        df = pd.read_csv(uploaded_file, sep='\t')
        if len(df.columns) < 2:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=r'\s+')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=None, engine='python') 
    
    # Force Data to Numeric (removes text headers/units)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)

    # --- NEW FEATURE: COLUMN SELECTORS ---
    st.sidebar.header("0. Select Data Columns")
    st.sidebar.markdown("Pick the column that actually contains the jagged anomaly (usually Load/Carico).")
    
    # Default X to Deformazione (Index 1) and Y to Carico (Index 0)
    col_deform = st.sidebar.selectbox("X-Axis (Deformation)", df.columns, index=1)
    col_stress = st.sidebar.selectbox("Y-Axis (Target for Detection)", df.columns, index=0)

    # --- Sidebar Controls ---
    st.sidebar.header("1. Target Slip Region")
    min_deform = st.sidebar.number_input("Min Deformation", value=float(df[col_deform].min()))
    max_deform = st.sidebar.number_input("Max Deformation", value=float(df[col_deform].max()))

    st.sidebar.header("2. Slip Detection Settings")
    z_thresh = st.sidebar.slider("Anomaly Sensitivity (Lower = Stricter)", 0.5, 10.0, 3.5)
    win_size = st.sidebar.slider("Analysis Window", 3, 51, 21, step=2) # Defaulted higher for your wide slips
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Deletion Expansion")
    dilation = st.sidebar.slider("Slip Width Erase (Points)", 0, 15, 3, 
                                 help="Increases the deletion zone to completely erase the 'crater' left by the machine slip.")

    st.sidebar.header("3. Cleaning Method")
    action = st.sidebar.radio("How to handle bad points:", ("Interpolate", "Delete Rows"), index=1)
    
    st.sidebar.header("4. Post-Processing")
    remove_break = st.sidebar.checkbox("✂️ Auto-Remove Breakpoint/Fracture", value=True)
    smooth_on = st.sidebar.checkbox("🌊 Apply Savitzky-Golay Smoothing", value=False)

    # Process Data
    df_cleaned, outliers = clean_tensile_data(
        df, col_deform, col_stress, min_deform, max_deform, 
        win_size, z_thresh, smooth_on, remove_break, action, dilation
    )

    # --- Top Visualization: Raw Data ---
    st.markdown("---")
    st.subheader(f"Raw Data: {col_stress} vs {col_deform}")
    fig_raw = go.Figure()
    
    fig_raw.add_trace(go.Scatter(x=df[col_deform], y=df[col_stress], name='Raw Data', line=dict(color='lightgrey', width=2)))
    fig_raw.add_trace(go.Scatter(x=outliers[col_deform], y=outliers[col_stress], mode='markers', name='Points Flagged for Deletion', marker=dict(color='red', size=8, symbol='x')))
    
    fig_raw.add_vline(x=min_deform, line_width=1, line_dash="dash", line_color="gray", annotation_text="Min Target")
    fig_raw.add_vline(x=max_deform, line_width=1, line_dash="dash", line_color="gray", annotation_text="Max Target")
    
    fig_raw.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), xaxis_title=col_deform, yaxis_title=col_stress)
    st.plotly_chart(fig_raw, use_container_width=True)

    # --- Bottom Visualization: Cleaned Data ---
    st.markdown("---")
    st.subheader("Final Cleaned Trend (Preview of Download)")
    fig_clean = go.Figure()
    
    fig_clean.add_trace(go.Scatter(x=df_cleaned[col_deform], y=df_cleaned[col_stress], name='Cleaned Data', line=dict(color='#003366', width=2)))
    
    fig_clean.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), xaxis_title=col_deform, yaxis_title=col_stress)
    st.plotly_chart(fig_clean, use_container_width=True)

    # --- Output & Export ---
    st.success(f"Processing complete! Original data rows: **{len(df)}**. Final cleaned rows: **{len(df_cleaned)}**. Total removed: **{len(df) - len(df_cleaned)}**.")

    # Prepare TXT (Tab Separated)
    csv = df_cleaned.to_csv(sep='\t', index=False).encode('utf-8')
    
    # Prepare Excel (XLSX)
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned Data')
    excel_data = excel_buffer.getvalue()

    # Download Buttons
    col1, col2 = st.columns([1, 1])

    with col1:
        st.download_button(
            label="Download Cleaned Data (TXT)",
            data=csv,
            file_name="cleaned_tensile_data.txt",
            mime="text/plain",
        )
        
    with col2:
        st.download_button(
            label="📊 Download Excel (XLSX)", 
            data=excel_data, 
            file_name="cleaned_tensile.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
