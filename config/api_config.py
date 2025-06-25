"""
Updated Fyers API Configuration with 5-minute timeframe support
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

# Data settings with ALL supported timeframes
DEFAULT_TICKERS = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:RELIANCE-EQ"]

# Updated to include 5-minute timeframe
DEFAULT_TIMEFRAMES = {
    "1min": "1",        # 1-minute
    "5min": "5",        # 5-minute (ADDED)
    "15min": "15",      # 15-minute
    "30min": "30",      # 30-minute
    "1hour": "60",      # 1-hour
    "4hour": "240",     # 4-hour
    "daily": "1D"       # Daily
}

# Timeframe display names for UI
TIMEFRAME_DISPLAY_NAMES = {
    "1min": "1 Minute",
    "5min": "5 Minutes",     # ADDED
    "15min": "15 Minutes", 
    "30min": "30 Minutes",
    "1hour": "1 Hour",
    "4hour": "4 Hours",
    "daily": "Daily"
}

# Timeframe categories for UI organization
TIMEFRAME_CATEGORIES = {
    "intraday_short": ["1min", "5min", "15min", "30min"],    # Added 5min here
    "intraday_medium": ["1hour", "4hour"], 
    "daily_plus": ["daily"]
}

# Data availability settings (for rate limiting and planning)
RATE_LIMIT_SETTINGS = {
    "safe_requests_per_second": 5,     # Conservative (50% of 10 limit)
    "safe_requests_per_minute": 150,   # Conservative (75% of 200 limit)
    "max_daily_requests": 100000,      # API daily limit
    "backoff_multiplier": 2.0,         # Exponential backoff
    "max_backoff_seconds": 60,         # Maximum wait time
    "max_retries": 3                   # Maximum retry attempts
}

# Bulk download settings
BULK_DOWNLOAD_DEFAULTS = {
    "batch_size": 5,                   # Process 5 symbols at a time
    "delay_between_batches": 2,        # 2 second delay between batches
    "daily_years_back": 5,             # Default years for daily data
    "intraday_years_back": 2,          # Default years for intraday data
    "max_concurrent_downloads": 3       # Max simultaneous downloads
}

# Path settings
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
SYMBOLS_PATH = os.path.join(DATA_PATH, "symbols")
CACHE_PATH = os.path.join(DATA_PATH, "cache")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# Create directories if they don't exist
for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, SYMBOLS_PATH, CACHE_PATH, RESULTS_PATH]:
    os.makedirs(path, exist_ok=True)

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard"
        },
        "file": {
            "level": "DEBUG", 
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join("logs", "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "detailed"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}