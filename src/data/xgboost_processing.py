import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_data_for_xgboost(file_path: str) -> pd.DataFrame:
    """
    Process data for XGBoost model.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        X: Input features as a DataFrame
        y: Target values as a Series
    """
    df = pd.read_csv(file_path)
    
    # Sort by user_id and assume sequential days
    # (Since we don't have explicit date, user_id + index order is assumed)
    # Reset index within group if needed but current file seems ordered by user
    
    # 1. Feature Engineering
    
    # Lag Features & Rolling
    # We need to process per user to avoid leaking data across users
    df_grouped = df.groupby('user_id')
    
    def apply_timeseries_features(group):
        # Shift 1 day
        group['distance_km_lag1'] = group['distance_km'].shift(1).fillna(0) # or bfill
        
        # Shift 7 days (Since only 7 days exist, this will be mostly NaN/0)
        # Using 0 or mean of valid data to be safe
        group['electricity_kwh_lag7'] = group['electricity_kwh'].shift(7).fillna(group['electricity_kwh'].mean())
        
        # Rolling 7 days
        # min_periods=1 accounts for start of sequence
        group['distance_km_rolling7'] = group['distance_km'].rolling(window=7, min_periods=1).mean()
        
        return group

    df = df_grouped.apply(apply_timeseries_features).reset_index(drop=True)
    
    # Categorical Encoding: One-Hot
    categorical_cols = ['transport_mode', 'food_type', 'day_type']
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    
    # Select features
    # Numerical features to scale
    numerical_cols = [
        'distance_km', 'electricity_kwh', 'renewable_usage_pct', 
        'screen_time_hours', 'waste_generated_kg', 'eco_actions',
        'distance_km_lag1', 'electricity_kwh_lag7', 'distance_km_rolling7'
    ]
    
    # 2. Standardization
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Prepare X and y
    target_col = 'carbon_footprint_kg'
    
    # Combine one-hot columns with numerical columns
    feature_cols = numerical_cols + [col for col in df.columns if col.startswith(tuple(categorical_cols))]
    
    X = df[feature_cols]
    y = df[target_col]
        
    print(f"XGBoost Data Processed: X shape={X.shape}, y shape={y.shape}")
    
    return X, y

if __name__ == "__main__":
    file_path = 'data/personal_carbon_footprint_behavior.csv'
    try:
        process_data_for_xgboost(file_path)
    except Exception as e:
        print(e)
