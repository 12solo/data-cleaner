import streamlit as st
import pandas as pd
import numpy as np

def remove_outliers_mad(df, col_stress, window_size=5, z_threshold=3.5):
    """Removes single-point anomalies using Robust Modified Z-Score (MAD)."""
    df_calc = df.copy()
    
    df_calc['Rolling_Median'] = df_calc[col_stress].rolling(window=window_size, center=True).median()
    df_calc['Rolling_Median'] = df_calc['Rolling_Median'].fillna(df_calc[col_stress])
    
    def calculate_mad(x):
        return np.median(np.abs(x - np.median(x)))
        
    df_calc['Rolling_MAD'] = df_calc[col_stress].rolling(window=window_size, center=True).apply(calculate_mad)
    df_calc['Rolling_MAD'] = df_calc['Rolling_MAD'].replace(0, 1e-6)
    
    df_calc['Mod_Z_Score'] = 0.6745 * np.abs(df_calc[col_stress] - df_calc['Rolling_Median']) / df_calc['Rolling_MAD']
    
    outliers = df_calc[df_calc['Mod_Z_Score'] > z_threshold].copy()
    df_cleaned = df_calc[df_calc['Mod_Z_Score'] <= z_threshold].copy()
    
    cols_to_drop = ['Rolling_Median', 'Rolling_MAD', 'Mod_Z_Score']
    outliers = outliers.drop(columns=cols_to_drop)
    df_cleaned = df_cleaned.drop(columns=cols_to_drop)
    
    return df_cleaned, outliers

# --- Streamlit UI ---
st.set_page_config(page_title="Tensile Data Cleaner", layout="centered")
st.title("Stress-Strain Anomaly Cleaner")
st.markdown("Upload your raw data. The robust MAD algorithm will silently process and clean the equipment drops.")

# File uploader
uploaded_file = st.file_uploader("Upload Raw Data (TXT/CSV)", type=['txt', 'csv'])

if uploaded_file:
    # Read the data
    df = pd.read_csv(uploaded_file, sep=r'\s+') 
    
    # Identify columns based on your structure
    col_deform = df.columns[1]
    col_stress = df.columns[2]

    # Sidebar parameters
    st.sidebar.header("Filter Settings")
    z_threshold = st.sidebar.slider("Anomaly Threshold", min_value=1.0, max_value=10.0, value=3.5, step=0.5)
    window_size = st.sidebar.slider("Rolling Window Size", min_value=3, max_value=15, value=5, step=2)

    # Process the data
    df_cleaned, outliers = remove_outliers_mad(df, col_stress, window_size, z_threshold)

    # Output Results
    st.success(f"Processing complete! The algorithm detected and removed **{len(outliers)}** outlier point(s).")

    # Download Button
    csv = df_cleaned.to_csv(sep='\t', index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data",
        data=csv,
        file_name="cleaned_tensile_data.txt",
        mime="text/plain",
    )
