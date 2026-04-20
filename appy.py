import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Cleaner", layout="wide")
st.title("Stress-Strain Curve Anomaly Cleaner")

st.markdown("Upload your raw data file to interactively isolate and remove the equipment error drop.")

# File uploader
uploaded_file = st.file_uploader("Upload Raw Data (TXT/CSV)", type=['txt', 'csv'])

if uploaded_file:
    # Read the uploaded file
    # Using regex separator \s+ to handle varying spaces and tabs in your raw data
    df = pd.read_csv(uploaded_file, sep=r'\s+') 
    
    # Identify columns based on your provided data structure
    col_load = df.columns[0]
    col_deform = df.columns[1]
    col_stress = df.columns[2]

    # Display Original Data Graph
    st.subheader("Original Data")
    fig_orig = px.line(df, x=col_deform, y=col_stress, title="Original Stress-Strain Curve")
    st.plotly_chart(fig_orig, use_container_width=True)

    # --- Filtering Section ---
    st.sidebar.header("Filter Settings")
    st.sidebar.markdown("Define the region of the equipment error.")
    
    # Interactive sliders/inputs to target the specific drop
    # Default values are set around the known anomaly at ~60mm
    min_deform = st.sidebar.number_input("Target Range: Min Deformation (mm)", value=55.0)
    max_deform = st.sidebar.number_input("Target Range: Max Deformation (mm)", value=65.0)
    
    # We want to drop points in this range that fall below the expected trend
    stress_threshold = st.sidebar.number_input("Stress Threshold (MPa) - drops below this will be removed", value=9.5)

    # Filtering Logic:
    # We keep rows that are OUTSIDE the target deformation range, 
    # OR if they are inside the range, their stress must be ABOVE the threshold.
    mask = ~((df[col_deform] >= min_deform) & (df[col_deform] <= max_deform) & (df[col_stress] < stress_threshold))
    df_cleaned = df[mask]

    # Display Cleaned Data Graph
    st.subheader("Cleaned Data")
    fig_clean = px.line(df_cleaned, x=col_deform, y=col_stress, title="Cleaned Stress-Strain Curve")
    fig_clean.update_traces(line=dict(color='green'))
    st.plotly_chart(fig_clean, use_container_width=True)

    # Download mechanism for the corrected data
    st.sidebar.markdown("---")
    csv = df_cleaned.to_csv(sep='\t', index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Cleaned Data",
        data=csv,
        file_name="corrected_data.txt",
        mime="text/plain",
    )
