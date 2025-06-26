"""
Quick fix for timeframe naming inconsistency
Run this once to standardize all folder names
"""
import os
import shutil

def standardize_timeframes():
    """Standardize timeframe folder names"""
    renames = {
        "15min": "15m",
        "daily": "1d", 
        "1hour": "1h"
    }
    
    for old_name, new_name in renames.items():
        for data_type in ["raw", "processed"]:
            old_path = f"data/{data_type}/{old_name}"
            new_path = f"data/{data_type}/{new_name}"
            
            if os.path.exists(old_path):
                if os.path.exists(new_path):
                    print(f"Merging {old_path} → {new_path}")
                    # Merge folders
                else:
                    print(f"Renaming {old_path} → {new_path}")
                    shutil.move(old_path, new_path)