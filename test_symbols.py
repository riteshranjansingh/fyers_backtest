"""
Test script for Symbol Master functionality - FIXED VERSION
"""
import logging
from src.data.symbol_master import SymbolMaster

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("\n📊 Testing Symbol Master (Fixed Version)...")
    print("-" * 50)
    
    # Create symbol master instance
    sm = SymbolMaster()
    
    # Step 1: Check current data and debug info
    print("\n🔍 Current Data Status:")
    debug_info = sm.get_debug_info()
    print(f"Status: {debug_info}")
    
    # Step 2: Load existing data first
    if not sm.load_symbols():
        print("\n❌ No cached symbols found. Need to download...")
        
        # Download fresh data
        print("\n📥 Downloading latest symbol master...")
        if sm.fetch_symbol_master("NSE"):
            print("✅ Symbol master downloaded successfully!")
        else:
            print("❌ Failed to download symbol master")
            return
    else:
        print("\n✅ Loaded cached symbol master")
        
        # Check if update is needed
        if sm.should_update():
            print("\n🔄 Data is outdated. Downloading fresh data...")
            if sm.fetch_symbol_master("NSE"):
                print("✅ Updated symbol master!")
            else:
                print("⚠️ Update failed, using cached data")
    
    # Step 3: Show debug information
    print("\n🔬 Debug Information:")
    print("-" * 50)
    debug_info = sm.get_debug_info()
    print(f"Total symbols: {debug_info.get('total_symbols', 0)}")
    print(f"Column mappings: {debug_info.get('column_mappings', {})}")
    
    # Show actual column structure
    columns = debug_info.get('columns', [])
    print(f"\nTotal columns: {len(columns)}")
    print("Key columns:")
    if len(columns) > 1:
        print(f"  Column 1 (names): {columns[1]}")
    if len(columns) > 9:
        print(f"  Column 9 (symbols): {columns[9]}")
    if len(columns) > 13:
        print(f"  Column 13 (short): {columns[13]}")
    
    # Show sample data from key columns
    if sm.symbols_df is not None and len(sm.symbols_df) > 0:
        print("\nSample data:")
        if len(columns) > 1:
            print(f"  Names sample: {list(sm.symbols_df.iloc[:3, 1])}")
        if len(columns) > 9:
            print(f"  Symbols sample: {list(sm.symbols_df.iloc[:3, 9])}")
        if len(columns) > 13:
            print(f"  Short sample: {list(sm.symbols_df.iloc[:3, 13])}")
    
    # Step 4: Show popular symbols
    print("\n🌟 Popular Symbols:")
    print("-" * 50)
    for symbol in sm.get_popular_symbols()[:5]:
        print(f"{symbol['symbol']:<25} - {symbol['name']}")
    
    # Step 5: Test search functionality
    print("\n🔍 Testing Symbol Search:")
    print("-" * 50)
    
    test_queries = ["RELIANCE", "NIFTY", "TCS", "HDFC", "BANK"]
    
    for query in test_queries:
        print(f"\nSearching for '{query}':")
        results = sm.search_symbols(query, limit=3)
        
        if results:
            for result in results:
                print(f"  {result['symbol']:<25} - {result['name']}")
        else:
            print("  No results found")
    
    # Step 6: Test specific symbol info
    print("\n📋 Testing Symbol Info:")
    print("-" * 50)
    test_symbol = "NSE:RELIANCE-EQ"
    info = sm.get_symbol_info(test_symbol)
    if info:
        print(f"Symbol: {info['symbol']}")
        print(f"Name: {info['name']}")
        print(f"ISIN: {info['isin']}")
    else:
        print(f"No info found for {test_symbol}")
    
    # Step 7: Interactive search (optional)
    print("\n" + "="*50)
    print("💡 Interactive Search (press Enter to skip):")
    while True:
        user_query = input("\nEnter symbol to search (or 'quit'): ").strip()
        
        if not user_query or user_query.lower() == 'quit':
            break
            
        results = sm.search_symbols(user_query, limit=5)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['symbol']:<25} - {result['name']}")
        else:
            print("No results found. Try another search term.")
    
    print("\n✅ Symbol master test complete!")

if __name__ == "__main__":
    main()