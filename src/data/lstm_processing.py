import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from typing import Tuple, List, Any

def process_data_for_lstm(file_path: str, window_size: int = 3, scaler_cls=MinMaxScaler) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Process data for LSTM/GRU models.
    
    Args:
        file_path: Path to the CSV file.
        window_size: Size of the sliding window.
        scaler_cls: Scaler class to use (MinMaxScaler or StandardScaler).
        
    Returns:
        X: Input features of shape (samples, window_size, features)
        y: Target values of shape (samples,)
        scaler: Fitted scaler for inverse transformation
    """
    df = pd.read_csv(file_path)
    
    # 1. Feature Engineering
    # Time related: day_type
    day_mapping = {'Weekday': 0, 'Weekend': 1, 'Holiday': 2}
    df['day_type_encoded'] = df['day_type'].map(day_mapping).fillna(0) # Default to 0 if unknown
    
    # Diet: food_type -> One-Hot
    food_dummies = pd.get_dummies(df['food_type'], prefix='food')
    df = pd.concat([df, food_dummies], axis=1)
    
    # Select features
    feature_cols = [
        'day_type_encoded', 
        'distance_km', 
        'electricity_kwh', 
        'renewable_usage_pct', 
        'screen_time_hours', 
        'waste_generated_kg', 
        'eco_actions'
    ]
    # Add one-hot encoded food columns
    feature_cols.extend(food_dummies.columns.tolist())
    
    target_col = 'carbon_footprint_kg'
    
    # Normalize features
    scaler = scaler_cls()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Target normalization (optional but often good for LSTM)
    # We will keep target original for interpretation or normalize it too? 
    # Usually better to normalize target for convergence. Let's normalize target separately.
    target_scaler = MinMaxScaler()
    df[[target_col]] = target_scaler.fit_transform(df[[target_col]])
    
    # 2. Sliding Window & Reshape
    X_list = []
    y_list = []
    
    # Group by user_id to avoid mixing data between users
    for user_id, group in df.groupby('user_id'):
        group = group.reset_index(drop=True)
        data_values = group[feature_cols].values
        target_values = group[target_col].values
        
        if len(group) <= window_size:
            continue
            
        for i in range(len(group) - window_size):
            X_list.append(data_values[i : i + window_size])
            y_list.append(target_values[i + window_size])
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"LSTM Data Processed: X shape={X.shape}, y shape={y.shape}")
    
    return X, y, target_scaler

if __name__ == "__main__":
    # Test
    file_path = 'data/personal_carbon_footprint_behavior.csv'
    try:
        process_data_for_lstm(file_path)
    except Exception as e:
        print(e)
