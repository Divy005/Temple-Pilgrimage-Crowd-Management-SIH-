import pandas as pd
import numpy as np
import os


if os.path.exists('X_train.csv') and os.path.exists('X_test.csv') and os.path.exists('y_train.csv') and os.path.exists('y_test.csv') and os.path.exists('dates_test.csv'):
    print("Split data already exists. Skipping preprocessing.")
else:
    print("Processing and splitting data...")
    
    df = pd.read_csv('final_data.csv')

    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    df['lag_1'] = df['darshans'].shift(1)
    df['lag_2'] = df['darshans'].shift(2)

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

 
    df = df.dropna()

    X = df.drop(['date', 'darshans'], axis=1) 
    y = df['darshans'] 
    dates = df['date'] 

 
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    dates_test = dates.iloc[train_size:]

  
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    pd.DataFrame({'darshans': y_train}).to_csv('y_train.csv', index=False)
    pd.DataFrame({'darshans': y_test}).to_csv('y_test.csv', index=False)
    pd.DataFrame({'date': dates_test}).to_csv('dates_test.csv', index=False)
    print("Split data saved as X_train.csv, X_test.csv, y_train.csv, y_test.csv, dates_test.csv")