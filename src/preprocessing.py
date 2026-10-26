import pandas as pd
import numpy as np
import re
import io
import os

def load_and_clean_data(file_path="data/india_housing_prices.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the dataset at {file_path}. Please make sure 'india_housing_prices.csv' is inside a folder named 'data'.")

    # 1. Read dataset handling character encodings safely
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    header_match = re.search(r'ID,State,City,Locality.*', text)
    if header_match:
        text = text[header_match.start():]

    df = pd.read_csv(io.StringIO(text))

    # Clean column names
    df.columns = [c.replace('\\', '').replace('_', ' ').strip().replace(' ', '_') for c in df.columns]

    if 'Locality' in df.columns:
        df['Locality'] = df['Locality'].str.replace('\\', '', regex=False)
    if 'Availability_Status' in df.columns:
        df['Availability_Status'] = df['Availability_Status'].str.replace('\\', '', regex=False)

    # Filter invalid records
    df = df[pd.to_numeric(df['ID'], errors='coerce').notnull()].copy()
    df['ID'] = df['ID'].astype(int)
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Convert Numeric Columns & Impute Missing Values
    numeric_cols = [
        'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 
        'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
        'Nearby_Schools', 'Nearby_Hospitals'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Impute categorical values with mode
    cat_cols = ['State', 'City', 'Locality', 'Property_Type', 'Furnished_Status', 
                'Public_Transport_Accessibility', 'Parking_Space', 'Security', 
                'Facing', 'Owner_Type', 'Availability_Status']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Create Target Variables (As per Project Guidelines)
    # Regression Target: Future Price after 5 Years (8% annual compound growth)
    annual_rate = 0.08
    df['Future_Price_5Y'] = (df['Price_in_Lakhs'] * ((1 + annual_rate) ** 5)).round(2)

    # Classification Target: Good Investment (Binary 0 or 1)
    city_medians = df.groupby('City')['Price_in_Lakhs'].transform('median')
    df['Good_Investment'] = np.where(
        (df['Price_in_Lakhs'] <= city_medians) & (df['BHK'] >= 2), 1, 0
    )

    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print("Preprocessing completed successfully!")
    print(f"Cleaned Dataset Shape: {df.shape}")
    print("\nSample Target Outputs:")
    print(df[['Price_in_Lakhs', 'Future_Price_5Y', 'Good_Investment']].head())
