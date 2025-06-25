"""
Data Inventory Manager
Tracks what data we have and what's missing
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config.api_config import RAW_DATA_PATH

class DataInventoryManager:
    """Manages inventory of downloaded data"""
    
    def __init__(self):
        self.inventory_file = os.path.join(RAW_DATA_PATH, "data_inventory.json")
        self.inventory = self._load_inventory()
    
    def _load_inventory(self) -> Dict:
        """Load existing inventory or create new one"""
        if os.path.exists(self.inventory_file):
            try:
                with open(self.inventory_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "last_updated": datetime.now().isoformat(),
            "symbols": {},
            "statistics": {
                "total_symbols": 0,
                "total_files": 0,
                "total_size_mb": 0
            }
        }
    
    def scan_data_directory(self):
        """Scan data directory and update inventory"""
        print("🔍 Scanning data directory...")
        
        symbols = {}
        total_files = 0
        total_size = 0
        
        # Scan each timeframe folder
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        
        for timeframe in timeframes:
            tf_path = os.path.join(RAW_DATA_PATH, timeframe)
            if not os.path.exists(tf_path):
                continue
                
            for filename in os.listdir(tf_path):
                if not filename.endswith('.csv'):
                    continue
                    
                file_path = os.path.join(tf_path, filename)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                # Parse filename: symbol_startdate_enddate.csv
                parts = filename.replace('.csv', '').split('_')
                if len(parts) >= 3:
                    symbol_parts = parts[:-2]
                    symbol = '_'.join(symbol_parts).replace('_', ':').replace(':', ':', 1).replace('_', '-')
                    start_date = parts[-2]
                    end_date = parts[-1]
                    
                    if symbol not in symbols:
                        symbols[symbol] = {}
                    
                    if timeframe not in symbols[symbol]:
                        symbols[symbol][timeframe] = []
                    
                    # Get record count
                    try:
                        df = pd.read_csv(file_path)
                        record_count = len(df)
                    except:
                        record_count = 0
                    
                    symbols[symbol][timeframe].append({
                        "filename": filename,
                        "start_date": start_date,
                        "end_date": end_date,
                        "records": record_count,
                        "size_mb": round(file_size, 2),
                        "last_modified": datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat()
                    })
                    
                    total_files += 1
                    total_size += file_size
        
        # Update inventory
        self.inventory = {
            "last_updated": datetime.now().isoformat(),
            "symbols": symbols,
            "statistics": {
                "total_symbols": len(symbols),
                "total_files": total_files,
                "total_size_mb": round(total_size, 2)
            }
        }
        
        self._save_inventory()
        print(f"✅ Inventory updated: {len(symbols)} symbols, {total_files} files")
    
    def _save_inventory(self):
        """Save inventory to file"""
        os.makedirs(os.path.dirname(self.inventory_file), exist_ok=True)
        with open(self.inventory_file, 'w') as f:
            json.dump(self.inventory, f, indent=2)
    
    def get_symbol_coverage(self, symbol: str) -> Dict:
        """Get data coverage for a symbol"""
        if symbol not in self.inventory["symbols"]:
            return {"symbol": symbol, "coverage": {}}
        
        coverage = {}
        symbol_data = self.inventory["symbols"][symbol]
        
        for timeframe, files in symbol_data.items():
            if files:
                # Find earliest and latest dates
                all_starts = [f["start_date"] for f in files]
                all_ends = [f["end_date"] for f in files]
                
                earliest = min(all_starts)
                latest = max(all_ends)
                total_records = sum(f["records"] for f in files)
                
                coverage[timeframe] = {
                    "earliest_date": earliest,
                    "latest_date": latest,
                    "total_records": total_records,
                    "file_count": len(files),
                    "total_size_mb": sum(f["size_mb"] for f in files)
                }
        
        return {"symbol": symbol, "coverage": coverage}
    
    def get_missing_data(self, symbol: str, timeframe: str, required_start: str, required_end: str) -> List[Dict]:
        """Identify missing data periods for a symbol"""
        coverage = self.get_symbol_coverage(symbol)
        
        if timeframe not in coverage["coverage"]:
            # No data at all
            return [{
                "start_date": required_start,
                "end_date": required_end,
                "reason": "No data available"
            }]
        
        tf_coverage = coverage["coverage"][timeframe]
        
        missing_periods = []
        
        # Check if we need data before earliest available
        if required_start < tf_coverage["earliest_date"]:
            missing_periods.append({
                "start_date": required_start,
                "end_date": tf_coverage["earliest_date"],
                "reason": "Data before earliest available"
            })
        
        # Check if we need data after latest available  
        if required_end > tf_coverage["latest_date"]:
            missing_periods.append({
                "start_date": tf_coverage["latest_date"],
                "end_date": required_end,
                "reason": "Data after latest available"
            })
        
        return missing_periods
    
    def print_inventory_summary(self):
        """Print a summary of the data inventory"""
        stats = self.inventory["statistics"]
        
        print("\n📊 DATA INVENTORY SUMMARY")
        print("=" * 50)
        print(f"📅 Last Updated: {self.inventory['last_updated']}")
        print(f"📊 Total Symbols: {stats['total_symbols']}")
        print(f"📁 Total Files: {stats['total_files']}")
        print(f"💾 Total Size: {stats['total_size_mb']:.2f} MB")
        
        print(f"\n📈 Top 10 Symbols by Data Volume:")
        print("-" * 40)
        
        # Calculate data volume per symbol
        symbol_volumes = {}
        for symbol, timeframes in self.inventory["symbols"].items():
            total_records = 0
            for tf, files in timeframes.items():
                total_records += sum(f["records"] for f in files)
            symbol_volumes[symbol] = total_records
        
        # Sort and show top 10
        top_symbols = sorted(symbol_volumes.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (symbol, records) in enumerate(top_symbols, 1):
            print(f"{i:2d}. {symbol:<25} {records:>10,} records")

# Test function
def test_inventory_manager():
    """Test the inventory manager"""
    print("🧪 Testing Data Inventory Manager")
    
    inventory = DataInventoryManager()
    inventory.scan_data_directory()
    inventory.print_inventory_summary()
    
    # Test coverage for specific symbols
    test_symbols = ["NSE:NIFTY50-INDEX", "NSE:RELIANCE-EQ"]
    
    for symbol in test_symbols:
        print(f"\n📊 Coverage for {symbol}:")
        coverage = inventory.get_symbol_coverage(symbol)
        
        if coverage["coverage"]:
            for timeframe, details in coverage["coverage"].items():
                print(f"  {timeframe}: {details['earliest_date']} to {details['latest_date']} ({details['total_records']:,} records)")
        else:
            print("  No data available")

if __name__ == "__main__":
    test_inventory_manager()