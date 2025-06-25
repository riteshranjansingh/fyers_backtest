"""
Quick script to check the structure of the downloaded symbol master CSV
"""
import pandas as pd
import os
from config.api_config import SYMBOLS_PATH

# Load the CSV
csv_file = os.path.join(SYMBOLS_PATH, "symbol_master.csv")
df = pd.read_csv(csv_file)

print("📊 Symbol Master CSV Structure:")
print("-" * 50)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print("\n📋 Column names:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

print("\n📄 First 5 rows:")
print(df.head())

print("\n🔍 Sample data for RELIANCE:")
# Try to find RELIANCE in different columns
for col in df.columns:
    if df[col].astype(str).str.contains('RELIANCE', case=False, na=False).any():
        print(f"\nFound 'RELIANCE' in column '{col}':")
        matches = df[df[col].astype(str).str.contains('RELIANCE', case=False, na=False)]
        print(matches.head(3))