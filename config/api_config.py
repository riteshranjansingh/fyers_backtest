"""
Fyers API Configuration using .env file
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment
FYERS_APP_ID = os.getenv("FYERS_APP_ID")
FYERS_APP_SECRET = os.getenv("FYERS_APP_SECRET")
FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
FYERS_REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "http://localhost:8080/callback")

# Validate that credentials are set
if not all([FYERS_APP_ID, FYERS_APP_SECRET, FYERS_CLIENT_ID]):
    print("ERROR: Missing Fyers credentials in .env file")
    print("Please check your .env file has all required values:")
    print("- FYERS_APP_ID")
    print("- FYERS_APP_SECRET") 
    print("- FYERS_CLIENT_ID")
    print("- FYERS_REDIRECT_URI")
    exit(1)

# API endpoints
BASE_URL = "https://api-t1.fyers.in/api/v3"
DATA_URL = "https://api-t1.fyers.in/data"

# Data settings
DEFAULT_TICKERS = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:RELIANCE-EQ"]
DEFAULT_TIMEFRAMES = {
    "15min": "15",
    "1hour": "60", 
    "4hour": "240",
    "daily": "1D"
}

# Path settings
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
SYMBOLS_PATH = os.path.join(DATA_PATH, "symbols")
CACHE_PATH = os.path.join(DATA_PATH, "cache")

# Create directories if they don't exist
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(SYMBOLS_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)
