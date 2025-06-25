"""
Test script for Symbol Master functionality
"""
import logging
from src.data.symbol_master import SymbolMaster

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("\n📊 Testing Symbol Master...")
    print("-" * 50)
    
    # Create symbol master instance
    sm = SymbolMaster()
    
    # Check if we need to update
    if sm.should_update():
        print("\n📥 Downloading latest symbol master...")
        if sm.fetch_symbol_master("NSE"):
            print("✅ Symbol master downloaded successfully!")
        else:
            print("❌ Failed to download symbol master")
            return
    else:
        print("\n✅ Using cached symbol master")
        sm.load_symbols()
    
    # Show popular symbols
    print("\n🌟 Popular Symbols:")
    print("-" * 50)
    for symbol in sm.get_popular_symbols()[:5]:
        print(f"{symbol['symbol']:<25} - {symbol['name']}")
    
    # Test search functionality
    print("\n🔍 Testing Symbol Search:")
    print("-" * 50)
    
    test_queries = ["RELIANCE", "NIFTY", "TCS", "HDFC"]
    
    for query in test_queries:
        print(f"\nSearching for '{query}':")
        results = sm.search_symbols(query, limit=3)
        
        if results:
            for result in results:
                print(f"  {result['symbol']:<25} - {result['name']}")
        else:
            print("  No results found")
    
    # Interactive search
    print("\n" + "="*50)
    print("💡 Try searching for any symbol (or press Enter to skip):")
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