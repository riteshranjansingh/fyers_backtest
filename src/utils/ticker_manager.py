"""
Comprehensive Ticker Management System
Supports selection, categorization, and management of multiple tickers for backtesting
"""
import json
import os
import pandas as pd
from typing import List, Dict, Optional, Set
from datetime import datetime
from src.data.symbol_master import SymbolMaster

class TickerManager:
    """Manages ticker selection, categorization, and validation for backtesting"""
    
    def __init__(self):
        self.symbol_master = SymbolMaster()
        self.ticker_lists_file = os.path.join("config", "ticker_lists.json")
        self.custom_lists = self._load_ticker_lists()
        
        # Predefined popular ticker categories
        self.predefined_lists = {
            "nifty_50_top_10": [
                "NSE:RELIANCE-EQ",
                "NSE:TCS-EQ", 
                "NSE:HDFCBANK-EQ",
                "NSE:INFY-EQ",
                "NSE:ICICIBANK-EQ",
                "NSE:HINDUNILVR-EQ",
                "NSE:SBIN-EQ",
                "NSE:BHARTIARTL-EQ",
                "NSE:ITC-EQ",
                "NSE:KOTAKBANK-EQ"
            ],
            "banking_stocks": [
                "NSE:HDFCBANK-EQ",
                "NSE:ICICIBANK-EQ", 
                "NSE:SBIN-EQ",
                "NSE:KOTAKBANK-EQ",
                "NSE:AXISBANK-EQ",
                "NSE:INDUSINDBK-EQ",
                "NSE:FEDERALBNK-EQ",
                "NSE:BANKBARODA-EQ"
            ],
            "it_stocks": [
                "NSE:TCS-EQ",
                "NSE:INFY-EQ",
                "NSE:WIPRO-EQ",
                "NSE:HCLTECH-EQ",
                "NSE:TECHM-EQ",
                "NSE:LTI-EQ",
                "NSE:MINDTREE-EQ"
            ],
            "pharma_stocks": [
                "NSE:SUNPHARMA-EQ",
                "NSE:DRREDDY-EQ",
                "NSE:CIPLA-EQ",
                "NSE:DIVISLAB-EQ",
                "NSE:BIOCON-EQ",
                "NSE:AUROPHARMA-EQ"
            ],
            "auto_stocks": [
                "NSE:MARUTI-EQ",
                "NSE:M&M-EQ",
                "NSE:TATAMOTORS-EQ",
                "NSE:BAJAJ-AUTO-EQ",
                "NSE:EICHERMOT-EQ",
                "NSE:HEROMOTOCO-EQ"
            ],
            "indices": [
                "NSE:NIFTY50-INDEX",
                "NSE:NIFTYBANK-INDEX",
                "NSE:NIFTYIT-INDEX",
                "NSE:NIFTYPHARMA-INDEX",
                "NSE:NIFTYAUTO-INDEX",
                "NSE:NIFTYFMCG-INDEX"
            ],
            "testing_sample": [
                "NSE:NIFTY50-INDEX",
                "NSE:RELIANCE-EQ",
                "NSE:TCS-EQ"
            ]
        }
    
    def _load_ticker_lists(self) -> Dict:
        """Load custom ticker lists from file"""
        if os.path.exists(self.ticker_lists_file):
            try:
                with open(self.ticker_lists_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_ticker_lists(self):
        """Save custom ticker lists to file"""
        os.makedirs(os.path.dirname(self.ticker_lists_file), exist_ok=True)
        with open(self.ticker_lists_file, 'w') as f:
            json.dump(self.custom_lists, f, indent=2)
    
    def get_available_lists(self) -> Dict[str, List[str]]:
        """Get all available ticker lists (predefined + custom)"""
        all_lists = {}
        all_lists.update(self.predefined_lists)
        all_lists.update(self.custom_lists)
        return all_lists
    
    def create_custom_list(self, name: str, tickers: List[str], description: str = "") -> bool:
        """Create a custom ticker list"""
        try:
            # Validate tickers
            valid_tickers = self.validate_tickers(tickers)
            invalid_tickers = set(tickers) - set(valid_tickers)
            
            if invalid_tickers:
                print(f"⚠️ Invalid tickers removed: {list(invalid_tickers)}")
            
            if not valid_tickers:
                print("❌ No valid tickers provided")
                return False
            
            self.custom_lists[name] = {
                "tickers": valid_tickers,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "count": len(valid_tickers)
            }
            
            self._save_ticker_lists()
            print(f"✅ Created custom list '{name}' with {len(valid_tickers)} tickers")
            return True
            
        except Exception as e:
            print(f"❌ Error creating custom list: {str(e)}")
            return False
    
    def search_tickers(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for tickers using symbol master"""
        if not self.symbol_master.load_symbols():
            print("❌ Could not load symbol master")
            return []
        
        return self.symbol_master.search_symbols(query, limit)
    
    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """Validate a list of tickers and return only valid ones"""
        if not self.symbol_master.load_symbols():
            print("⚠️ Could not load symbol master for validation")
            return tickers  # Return as-is if can't validate
        
        valid_tickers = []
        for ticker in tickers:
            if self.symbol_master.validate_symbol(ticker):
                valid_tickers.append(ticker)
            else:
                print(f"⚠️ Invalid ticker: {ticker}")
        
        return valid_tickers
    
    def get_ticker_list(self, list_name: str) -> List[str]:
        """Get tickers from a specific list"""
        all_lists = self.get_available_lists()
        
        if list_name in all_lists:
            if isinstance(all_lists[list_name], list):
                return all_lists[list_name]
            elif isinstance(all_lists[list_name], dict):
                return all_lists[list_name].get("tickers", [])
        
        print(f"❌ List '{list_name}' not found")
        return []
    
    def get_tickers_by_category(self, category: str) -> List[str]:
        """Get tickers by category (banking, it, pharma, etc.)"""
        category_key = f"{category.lower()}_stocks"
        if category_key in self.predefined_lists:
            return self.predefined_lists[category_key]
        
        # Search in custom lists
        for name, data in self.custom_lists.items():
            if category.lower() in name.lower():
                if isinstance(data, dict):
                    return data.get("tickers", [])
                return data
        
        return []
    
    def combine_lists(self, list_names: List[str], remove_duplicates: bool = True) -> List[str]:
        """Combine multiple ticker lists"""
        combined = []
        
        for list_name in list_names:
            tickers = self.get_ticker_list(list_name)
            combined.extend(tickers)
        
        if remove_duplicates:
            combined = list(dict.fromkeys(combined))  # Preserve order while removing duplicates
        
        return combined
    
    def filter_tickers_by_availability(
        self, 
        tickers: List[str], 
        timeframe: str,
        min_years: int = 1
    ) -> Dict[str, List[str]]:
        """Filter tickers based on data availability"""
        from src.data.fetcher import RateLimitSafeDataFetcher
        
        fetcher = RateLimitSafeDataFetcher()
        available = []
        unavailable = []
        limited = []
        
        print(f"🔍 Checking data availability for {len(tickers)} tickers...")
        
        for ticker in tickers:
            try:
                availability = fetcher.get_symbol_availability(ticker)
                
                if timeframe == "1d":
                    start_date_str = availability["daily_start"]
                else:
                    start_date_str = availability["intraday_start"]
                
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                years_available = (datetime.now() - start_date).days / 365.25
                
                if years_available >= min_years:
                    available.append(ticker)
                else:
                    limited.append(ticker)
                    
            except Exception as e:
                unavailable.append(ticker)
                print(f"⚠️ Could not check availability for {ticker}: {str(e)}")
        
        return {
            "available": available,
            "limited_data": limited,
            "unavailable": unavailable
        }
    
    def get_ticker_statistics(self, tickers: List[str]) -> Dict:
        """Get statistics about a list of tickers"""
        if not tickers:
            return {"error": "No tickers provided"}
        
        stats = {
            "total_count": len(tickers),
            "unique_count": len(set(tickers)),
            "duplicates": len(tickers) - len(set(tickers)),
            "by_exchange": {},
            "by_type": {},
            "invalid_format": []
        }
        
        for ticker in set(tickers):  # Use set to count unique only
            if ":" not in ticker:
                stats["invalid_format"].append(ticker)
                continue
            
            try:
                exchange, symbol_part = ticker.split(":", 1)
                
                # Count by exchange
                stats["by_exchange"][exchange] = stats["by_exchange"].get(exchange, 0) + 1
                
                # Determine type
                if "INDEX" in symbol_part:
                    instrument_type = "INDEX"
                elif "-EQ" in symbol_part:
                    instrument_type = "EQUITY"
                elif "-FUT" in symbol_part:
                    instrument_type = "FUTURES"
                else:
                    instrument_type = "OTHER"
                
                stats["by_type"][instrument_type] = stats["by_type"].get(instrument_type, 0) + 1
                
            except Exception:
                stats["invalid_format"].append(ticker)
        
        return stats
    
    def suggest_tickers(self, base_ticker: str, count: int = 5) -> List[str]:
        """Suggest similar tickers based on a base ticker"""
        suggestions = []
        
        # If it's an index, suggest related indices
        if "INDEX" in base_ticker:
            suggestions.extend(self.predefined_lists["indices"])
        
        # If it's a stock, suggest from same sector
        elif any(base_ticker in sector_list for sector_list in [
            self.predefined_lists["banking_stocks"],
            self.predefined_lists["it_stocks"],
            self.predefined_lists["pharma_stocks"],
            self.predefined_lists["auto_stocks"]
        ]):
            # Find which sector and suggest from that sector
            for sector, tickers in [
                ("banking", self.predefined_lists["banking_stocks"]),
                ("it", self.predefined_lists["it_stocks"]),
                ("pharma", self.predefined_lists["pharma_stocks"]),
                ("auto", self.predefined_lists["auto_stocks"])
            ]:
                if base_ticker in tickers:
                    suggestions.extend(tickers)
                    break
        
        # Remove the base ticker and limit results
        suggestions = [t for t in suggestions if t != base_ticker]
        return suggestions[:count]
    
    def print_available_lists(self):
        """Print all available ticker lists with details"""
        print("\n📊 AVAILABLE TICKER LISTS")
        print("=" * 50)
        
        print("\n🏗️ Predefined Lists:")
        for name, tickers in self.predefined_lists.items():
            print(f"  • {name}: {len(tickers)} tickers")
        
        if self.custom_lists:
            print("\n👤 Custom Lists:")
            for name, data in self.custom_lists.items():
                if isinstance(data, dict):
                    count = data.get("count", len(data.get("tickers", [])))
                    desc = data.get("description", "")
                    print(f"  • {name}: {count} tickers - {desc}")
                else:
                    print(f"  • {name}: {len(data)} tickers")
        
        print(f"\n📈 Total Lists: {len(self.predefined_lists) + len(self.custom_lists)}")

# Test function for ticker manager
def test_ticker_manager():
    """Test the ticker manager functionality"""
    print("🧪 Testing Ticker Manager")
    
    tm = TickerManager()
    
    # Show available lists
    tm.print_available_lists()
    
    # Test getting a specific list
    print(f"\n📋 Banking stocks: {tm.get_ticker_list('banking_stocks')}")
    
    # Test ticker search
    print(f"\n🔍 Search results for 'BANK':")
    results = tm.search_tickers("BANK", limit=5)
    for result in results:
        print(f"  {result['symbol']} - {result['name']}")
    
    # Test creating custom list
    custom_tickers = ["NSE:WIPRO-EQ", "NSE:HCLTECH-EQ", "NSE:TECHM-EQ"]
    tm.create_custom_list("my_it_stocks", custom_tickers, "My favorite IT stocks")
    
    # Test combining lists
    combined = tm.combine_lists(["banking_stocks", "it_stocks"])
    print(f"\n🔗 Combined banking + IT: {len(combined)} tickers")
    
    # Test ticker statistics
    stats = tm.get_ticker_statistics(combined)
    print(f"\n📊 Combined list statistics:")
    print(f"   Total: {stats['total_count']}")
    print(f"   By exchange: {stats['by_exchange']}")
    print(f"   By type: {stats['by_type']}")
    
    return tm

if __name__ == "__main__":
    test_ticker_manager()