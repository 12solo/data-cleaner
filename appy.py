import numpy as np
import pandas as pd

def remove_outliers_mad(df, col_stress, window_size=5, z_threshold=3.5):
    """
    Removes single-point anomalies from a dataset using the Robust Modified Z-Score (MAD).
    
    Parameters:
    - df: pandas DataFrame containing the test data.
    - col_stress: string, the name of the column containing stress values.
    - window_size: int, number of points to calculate the local trend (default 5).
    - z_threshold: float, the statistical threshold for anomaly detection (default 3.5).
    
    Returns:
    - df_cleaned: DataFrame with the outliers removed.
    - outliers: DataFrame containing only the detected outlier rows.
    """
    # Work on a copy to avoid modifying the original dataframe
    df_calc = df.copy()
    
    # 1. Calculate the rolling median (the robust local trend)
    df_calc['Rolling_Median'] = df_calc[col_stress].rolling(window=window_size, center=True).median()
    df_calc['Rolling_Median'] = df_calc['Rolling_Median'].fillna(df_calc[col_stress])
    
    # 2. Calculate the rolling Median Absolute Deviation (MAD)
    def calculate_mad(x):
        return np.median(np.abs(x - np.median(x)))
        
    df_calc['Rolling_MAD'] = df_calc[col_stress].rolling(window=window_size, center=True).apply(calculate_mad)
    
    # Prevent division by zero in perfectly flat areas
    df_calc['Rolling_MAD'] = df_calc['Rolling_MAD'].replace(0, 1e-6)
    
    # 3. Calculate Modified Z-Score
    df_calc['Mod_Z_Score'] = 0.6745 * np.abs(df_calc[col_stress] - df_calc['Rolling_Median']) / df_calc['Rolling_MAD']
    
    # 4. Separate the good data from the outliers
    outliers = df_calc[df_calc['Mod_Z_Score'] > z_threshold].copy()
    df_cleaned = df_calc[df_calc['Mod_Z_Score'] <= z_threshold].copy()
    
    # 5. Clean up the temporary calculation columns
    cols_to_drop = ['Rolling_Median', 'Rolling_MAD', 'Mod_Z_Score']
    outliers = outliers.drop(columns=cols_to_drop)
    df_cleaned = df_cleaned.drop(columns=cols_to_drop)
    
    return df_cleaned, outliers
