"""
Test script for Fyers API connection
Run this to verify your API setup is working correctly
"""
import logging
from src.api.connection import FyersConnection

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("\n🚀 Testing Fyers API Connection...")
    print("-" * 50)
    
    # Create connection instance
    conn = FyersConnection()
    
    # Try to connect
    if conn.connect():
        print("\n✅ Connection successful!")
        
        # Test connection and show account info
        account_info = conn.test_connection()
        if account_info['status'] == 'success':
            print(f"\n📊 Account Information:")
            print(f"   Name: {account_info['user_name']}")
            print(f"   User ID: {account_info['user_id']}")
            print(f"   Email: {account_info['email']}")
            print(f"   Broker: {account_info['broker']}")
        
        # Get funds information
        funds_info = conn.get_funds()
        if funds_info['status'] == 'success':
            print(f"\n💰 Funds Information:")
            print(f"   Balance: ₹{funds_info['balance']:,.2f}")
            print(f"   Available: ₹{funds_info['available']:,.2f}")
            print(f"   Utilized: ₹{funds_info['utilized']:,.2f}")
            
        print("\n✅ All tests passed! Your Fyers API is ready to use.")
    else:
        print("\n❌ Connection failed. Please check your credentials and try again.")
        print("\nCommon issues:")
        print("1. Check your .env file has correct credentials")
        print("2. Make sure your app is active in Fyers API portal")
        print("3. Verify redirect URI matches exactly")

if __name__ == "__main__":
    main()