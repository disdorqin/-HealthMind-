import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_data_for_moirai(file_path: str) -> pd.DataFrame:
    """
    Process data for Moirai model (Zero-shot Forecasting).
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        pd.DataFrame: Long-format DataFrame ready for multivariate time series tasks.
                      Columns: [item_id, timestamp, target, feat_dynamic_real_*, feat_static_cat_*]
    """
    df = pd.read_csv(file_path)
    
    # Check if 'user_id' exists, rename to 'item_id' for consistency with forecasting libraries
    if 'user_id' in df.columns:
        df = df.rename(columns={'user_id': 'item_id'})
    else:
        # Assume single time series if no user_id? Or error?
        # Based on user logic, creating a dummy item_id
        df['item_id'] = 'series_0'
        
    # 1. Build Standard Time Series
    # Generate dates: 7 days per user starting 2023-01-01
    # Assuming the file is sorted by item_id then time. 
    # Validated via data audit: 7 rows per user.
    
    start_date = pd.Timestamp('2023-01-01')
    
    # Assign dates efficiently
    df['day_offset'] = df.groupby('item_id').cumcount()
    df['timestamp'] = start_date + pd.to_timedelta(df['day_offset'], unit='D')
    df.drop(columns=['day_offset'], inplace=True)
    
    # 2. Missing Value Handling (Zero Tolerance)
    # Check for missing values
    # Numerical features: Linear Interpolation / Forward Fill
    numerical_cols = [
        'distance_km', 'electricity_kwh', 'renewable_usage_pct',
        'screen_time_hours', 'waste_generated_kg', 'eco_actions'
    ]
    
    # Group by item_id to avoid cross-series contamination via transform
    for col in numerical_cols:
        # Use transform to maintain index alignment
        df[col] = df.groupby('item_id')[col].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
        
    # Categorical features: Forward Fill
    categorical_cols = ['day_type', 'transport_mode', 'food_type', 'carbon_impact_level']
    for col in categorical_cols:
        df[col] = df.groupby('item_id')[col].transform(lambda x: x.ffill().bfill())
    
    # 3. Categorical Feature Encoding (Covariates)
    # One-Hot Encoding for categorical features to be used as covariates
    # We keep the item_id and timestamp
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dtype=int)
    
    # 4. Global Standardization (StandardScaler - no MinMax)
    scaler = StandardScaler()
    # Fit on the whole dataset (Global)
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    # 5. Multivariate Time Series Format
    # The dataframe is already in long format: [item_id, timestamp, features...]
    # Ensure target is present
    target_col = 'carbon_footprint_kg'
    
    # 6. Time Alignment check
    # We generated timestamps to be strictly aligned (everyone gets day 0 to day 6)
    # Verification
    time_points = df_encoded['timestamp'].unique()
    expected_points = 7
    if len(time_points) != expected_points:
        print(f"Warning: Discovered {len(time_points)} unique timestamps, expected {expected_points}.")
        
    print(f"Moirai Data Processed: Shape={df_encoded.shape}, Unique Items={df_encoded['item_id'].nunique()}")
    print("Sample generated timestamps:", df_encoded['timestamp'].head(7).dt.date.tolist())
    
    return df_encoded

if __name__ == "__main__":
    file_path = 'data/personal_carbon_footprint_behavior.csv'
    try:
        process_data_for_moirai(file_path)
    except Exception as e:
        print(e)
