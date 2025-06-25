"""
Rate Limit Safe Data Fetcher
Enhanced with proper rate limiting and error handling
"""
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Tuple
from tqdm import tqdm
import time
import random

from src.api.connection import FyersConnection
from config.api_config import RAW_DATA_PATH, PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)


class RateLimitSafeDataFetcher:
    """
    Rate limit safe data fetcher with proper error handling and backoff
    """
    
    def __init__(self, connection: FyersConnection = None):
        self.connection = connection or FyersConnection()
        self.fyers = None
        
        # Ensure data directories exist
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        
        # Conservative rate limiting (well below API limits)
        self.SAFE_REQUESTS_PER_SECOND = 5    # 50% of limit (10)
        self.SAFE_REQUESTS_PER_MINUTE = 150  # 75% of limit (200)
        self.SAFE_DELAY_BETWEEN_REQUESTS = 1.0 / self.SAFE_REQUESTS_PER_SECOND  # 200ms
        
        # API limitations based on official Fyers documentation
        self.MAX_DAYS_INTRADAY = 100    # For 1m to 240m resolutions
        self.MAX_DAYS_DAILY = 366       # For 1D resolution
        
        # Rate limiting tracking
        self.request_timestamps = []
        self.rate_limit_violations = 0
        self.last_rate_limit_reset = datetime.now()
        
        # Retry and backoff settings
        self.MAX_RETRIES = 3
        self.BASE_BACKOFF_DELAY = 2.0  # 2 seconds base delay
        self.MAX_BACKOFF_DELAY = 60.0  # 1 minute max delay
        
        # Data availability mapping
        self.DATA_AVAILABILITY = {
            "NSE:NIFTY50-INDEX": {
                "daily_start": "2000-01-01",
                "intraday_start": "2018-01-01"
            },
            "NSE:NIFTYBANK-INDEX": {
                "daily_start": "2006-01-01", 
                "intraday_start": "2018-01-01"
            },
            "default_stock": {
                "daily_start": "2000-01-01",
                "intraday_start": "2017-07-03"
            }
        }
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within safe rate limits"""
        now = datetime.now()
        
        # Clean old timestamps (older than 1 minute)
        minute_ago = now - timedelta(minutes=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        # Check per-minute limit
        if len(self.request_timestamps) >= self.SAFE_REQUESTS_PER_MINUTE:
            logger.warning(f"⚠️ Approaching per-minute rate limit ({len(self.request_timestamps)}/{self.SAFE_REQUESTS_PER_MINUTE})")
            return False
        
        # Check per-second limit (last 1 second)
        second_ago = now - timedelta(seconds=1)
        recent_requests = [ts for ts in self.request_timestamps if ts > second_ago]
        
        if len(recent_requests) >= self.SAFE_REQUESTS_PER_SECOND:
            logger.warning(f"⚠️ Approaching per-second rate limit ({len(recent_requests)}/{self.SAFE_REQUESTS_PER_SECOND})")
            return False
        
        return True
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits"""
        if not self._check_rate_limits():
            # Calculate wait time based on oldest request in current minute
            if self.request_timestamps:
                oldest_request = min(self.request_timestamps)
                wait_time = 60 - (datetime.now() - oldest_request).total_seconds()
                wait_time = max(wait_time, 1.0)  # At least 1 second
                
                logger.info(f"⏸️ Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # Always add base delay between requests
        time.sleep(self.SAFE_DELAY_BETWEEN_REQUESTS)
        
        # Add small random jitter to avoid synchronized requests
        jitter = random.uniform(0, 0.1)
        time.sleep(jitter)
    
    def _record_request(self):
        """Record a request timestamp for rate limiting"""
        self.request_timestamps.append(datetime.now())
    
    def _handle_api_response(self, response: Dict, context: str = "") -> Tuple[bool, Optional[str]]:
        """
        Handle API response and check for errors
        Returns: (success, error_message)
        """
        if not response:
            return False, "Empty response"
        
        # Check for successful response
        if response.get("s") == "ok":
            return True, None
        
        # Handle error responses
        error_code = response.get("code")
        error_message = response.get("message", "Unknown error")
        
        # Rate limit errors
        if error_code == 429 or error_code == -429:
            self.rate_limit_violations += 1
            logger.error(f"🚨 Rate limit exceeded! Violation #{self.rate_limit_violations}")
            
            if self.rate_limit_violations >= 3:
                logger.critical("💀 CRITICAL: 3 rate limit violations! User will be blocked for the day!")
                return False, "Daily rate limit violations exceeded - user blocked"
            
            return False, f"Rate limit exceeded (violation #{self.rate_limit_violations})"
        
        # Invalid symbol
        if error_code == -300:
            return False, f"Invalid symbol: {error_message}"
        
        # Token errors
        if error_code in [-8, -15, -16, -17]:
            return False, f"Authentication error: {error_message}"
        
        # Other errors
        return False, f"API error {error_code}: {error_message}"
    
    def _fetch_with_retry(self, params: Dict, context: str = "") -> Optional[pd.DataFrame]:
        """Fetch data with retry logic and rate limiting"""
        
        for attempt in range(self.MAX_RETRIES):
            try:
                # Wait for rate limits
                self._wait_for_rate_limit()
                
                # Record the request
                self._record_request()
                
                # Make API call
                response = self.fyers.history(data=params)
                
                # Handle response
                success, error_msg = self._handle_api_response(response, context)
                
                if success and "candles" in response:
                    # Process successful response
                    df = pd.DataFrame(
                        response["candles"],
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    
                    if not df.empty:
                        # Convert timestamp and set as index
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                        df.set_index("timestamp", inplace=True)
                        
                        # Ensure numeric columns
                        for col in ["open", "high", "low", "close", "volume"]:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        return df
                
                elif not success:
                    # Handle specific errors
                    if "rate limit" in error_msg.lower():
                        # Exponential backoff for rate limits
                        backoff_delay = min(
                            self.BASE_BACKOFF_DELAY * (2 ** attempt),
                            self.MAX_BACKOFF_DELAY
                        )
                        logger.warning(f"⏳ Rate limit hit. Backing off for {backoff_delay:.1f}s (attempt {attempt + 1})")
                        time.sleep(backoff_delay)
                        continue
                    
                    elif "invalid symbol" in error_msg.lower():
                        # Don't retry for invalid symbols
                        logger.error(f"❌ {context}: {error_msg}")
                        return pd.DataFrame()
                    
                    elif "authentication" in error_msg.lower():
                        # Try to reconnect
                        logger.warning(f"🔄 Authentication issue. Attempting reconnection...")
                        if self.connect():
                            continue
                        else:
                            logger.error(f"❌ Failed to reconnect: {error_msg}")
                            return pd.DataFrame()
                    
                    else:
                        logger.warning(f"⚠️ {context}: {error_msg} (attempt {attempt + 1})")
                
                # If we get here, retry with exponential backoff
                if attempt < self.MAX_RETRIES - 1:
                    backoff_delay = min(
                        self.BASE_BACKOFF_DELAY * (2 ** attempt),
                        self.MAX_BACKOFF_DELAY
                    )
                    time.sleep(backoff_delay)
                
            except Exception as e:
                logger.error(f"❌ {context}: Exception on attempt {attempt + 1}: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.BASE_BACKOFF_DELAY * (2 ** attempt))
                else:
                    logger.error(f"💀 {context}: All retry attempts failed")
        
        return pd.DataFrame()
    
    def connect(self) -> bool:
        """Ensure API connection is established"""
        if not self.fyers:
            if self.connection.connect():
                self.fyers = self.connection.get_session()
                return True
            return False
        return True
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        years_back: int = None,
        save_to_file: bool = True,
        force_refresh: bool = False,
        validate_dates: bool = True
    ) -> pd.DataFrame:
        """
        Get historical data with enhanced rate limiting and error handling
        """
        if not self.connect():
            logger.error("Failed to connect to Fyers API")
            return pd.DataFrame()
        
        # Handle date parameters
        end_date = self._parse_date(end_date) or datetime.now()
        
        if start_date:
            start_date = self._parse_date(start_date)
        elif years_back:
            start_date = end_date - timedelta(days=years_back * 365)
        else:
            start_date, end_date = self.get_optimal_date_range(symbol, timeframe)
        
        # Validate and adjust dates if requested
        warnings = []
        if validate_dates:
            start_date, end_date, warnings = self.validate_and_adjust_dates(
                symbol, timeframe, start_date, end_date
            )
            
            for warning in warnings:
                print(warning)
                logger.warning(warning)
        
        # Check for cached data first
        if not force_refresh:
            cached_data = self._load_cached_data(symbol, timeframe, start_date, end_date)
            if cached_data is not None and not cached_data.empty:
                logger.info(f"✅ Using cached data for {symbol} ({len(cached_data)} records)")
                return cached_data
        
        # Calculate total days needed
        total_days = (end_date - start_date).days
        max_days_per_request = self._get_max_days_for_timeframe(timeframe)
        
        logger.info(f"📊 Fetching {total_days} days of {timeframe} data for {symbol}")
        logger.info(f"🛡️ Rate limiting: {self.SAFE_REQUESTS_PER_SECOND} req/sec, {self.SAFE_REQUESTS_PER_MINUTE} req/min")
        
        if total_days <= max_days_per_request:
            # Single request
            data = self._fetch_single_chunk(symbol, timeframe, start_date, end_date)
        else:
            # Multiple requests with chunking
            data = self._fetch_with_chunking(symbol, timeframe, start_date, end_date)
        
        # Save to file if requested
        if save_to_file and not data.empty:
            self._save_data(data, symbol, timeframe, start_date, end_date)
        
        return data
    
    def _fetch_single_chunk(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data for a single time period with rate limiting"""
        params = {
            "symbol": symbol,
            "resolution": self._map_timeframe(timeframe),
            "date_format": "1",
            "range_from": start_date.strftime("%Y-%m-%d"),
            "range_to": end_date.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        
        context = f"{symbol} {timeframe} {start_date.date()} to {end_date.date()}"
        df = self._fetch_with_retry(params, context)
        
        if not df.empty:
            logger.info(f"✅ Fetched {len(df)} records for {context}")
        else:
            logger.warning(f"❌ No data for {context}")
        
        return df
    
    def _fetch_with_chunking(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data in chunks with enhanced rate limiting"""
        logger.info(f"🔄 Using chunked fetching for {symbol}")
        
        max_days_per_request = self._get_max_days_for_timeframe(timeframe)
        
        all_chunks = []
        current_end = end_date
        chunk_number = 1
        
        # Calculate total chunks for progress bar
        total_days = (end_date - start_date).days
        total_chunks = (total_days // max_days_per_request) + 1
        
        logger.info(f"📦 Using {max_days_per_request}-day chunks for {timeframe} data ({total_chunks} chunks needed)")
        
        # Estimate total time with rate limiting
        estimated_time = total_chunks * self.SAFE_DELAY_BETWEEN_REQUESTS / 60  # minutes
        logger.info(f"⏱️ Estimated time with rate limiting: {estimated_time:.1f} minutes")
        
        with tqdm(total=total_chunks, desc=f"Fetching {symbol}", unit="chunk") as pbar:
            while current_end > start_date:
                chunk_start = max(start_date, current_end - timedelta(days=max_days_per_request))
                
                pbar.set_description(f"Chunk {chunk_number}/{total_chunks}: {chunk_start.date()} to {current_end.date()}")
                
                chunk_data = self._fetch_single_chunk(symbol, timeframe, chunk_start, current_end)
                
                if not chunk_data.empty:
                    all_chunks.append(chunk_data)
                    logger.debug(f"Chunk {chunk_number}: {len(chunk_data)} records")
                else:
                    logger.warning(f"Empty chunk {chunk_number}")
                
                # Move to next chunk
                current_end = chunk_start - timedelta(days=1)
                chunk_number += 1
                pbar.update(1)
        
        # Combine all chunks
        if all_chunks:
            combined_data = pd.concat(all_chunks)
            combined_data = combined_data.sort_index()
            
            # Remove duplicates (can happen at chunk boundaries)
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            
            logger.info(f"✅ Combined {len(combined_data)} total records from {len(all_chunks)} chunks")
            return combined_data
        else:
            logger.error("No data retrieved from any chunk")
            return pd.DataFrame()
    
    def get_rate_limit_status(self) -> Dict:
        """Get current rate limiting status"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        second_ago = now - timedelta(seconds=1)
        
        recent_minute = [ts for ts in self.request_timestamps if ts > minute_ago]
        recent_second = [ts for ts in self.request_timestamps if ts > second_ago]
        
        return {
            "requests_last_minute": len(recent_minute),
            "requests_last_second": len(recent_second),
            "rate_limit_violations": self.rate_limit_violations,
            "safe_requests_per_minute": self.SAFE_REQUESTS_PER_MINUTE,
            "safe_requests_per_second": self.SAFE_REQUESTS_PER_SECOND,
            "current_delay": self.SAFE_DELAY_BETWEEN_REQUESTS
        }
    
    # ... (include all other methods from previous fetcher with same implementations)
    
    def _get_max_days_for_timeframe(self, timeframe: str) -> int:
        """Get maximum days per request based on timeframe"""
        if timeframe == "1d":
            return self.MAX_DAYS_DAILY  # 366 days for daily
        else:
            return self.MAX_DAYS_INTRADAY  # 100 days for intraday
    
    def _map_timeframe(self, timeframe: str) -> str:
        """Map user-friendly timeframe to Fyers API format"""
        mapping = {
            "1m": "1",
            "5m": "5",      # Added 5-minute support
            "15m": "15",
            "30m": "30",
            "1h": "60", 
            "4h": "240",
            "1d": "1D",
            "1w": "1W",
            "1M": "1M"
        }
        return mapping.get(timeframe, timeframe)
    
    # ... (include other helper methods: _parse_date, _load_cached_data, _save_data, etc.)
    
    def get_symbol_availability(self, symbol: str) -> Dict[str, str]:
        """Get data availability dates for a specific symbol"""
        if symbol in self.DATA_AVAILABILITY:
            return self.DATA_AVAILABILITY[symbol]
        if "INDEX" in symbol:
            return self.DATA_AVAILABILITY["NSE:NIFTY50-INDEX"]
        return self.DATA_AVAILABILITY["default_stock"]
    
    def validate_and_adjust_dates(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Tuple[datetime, datetime, List[str]]:
        """Validate and adjust dates based on data availability"""
        warnings = []
        availability = self.get_symbol_availability(symbol)
        
        if timeframe == "1d":
            earliest_available = datetime.strptime(availability["daily_start"], "%Y-%m-%d")
            data_type = "daily"
        else:
            earliest_available = datetime.strptime(availability["intraday_start"], "%Y-%m-%d")
            data_type = "intraday"
        
        original_start = start_date
        if start_date < earliest_available:
            start_date = earliest_available
            warnings.append(
                f"⚠️ {symbol} {data_type} data only available from {earliest_available.date()}. "
                f"Adjusted from {original_start.date()} to {start_date.date()}."
            )
        
        today = datetime.now().date()
        if end_date.date() > today:
            end_date = datetime.combine(today, datetime.min.time())
            warnings.append(f"⚠️ End date adjusted to today ({today}) as future data not available.")
        
        return start_date, end_date, warnings
    
    def get_optimal_date_range(self, symbol: str, timeframe: str, years_back: int = None) -> Tuple[datetime, datetime]:
        """Get optimal date range for backtesting based on data availability"""
        availability = self.get_symbol_availability(symbol)
        end_date = datetime.now()
        
        if timeframe == "1d":
            earliest_start = datetime.strptime(availability["daily_start"], "%Y-%m-%d")
        else:
            earliest_start = datetime.strptime(availability["intraday_start"], "%Y-%m-%d")
        
        if years_back:
            requested_start = end_date - timedelta(days=years_back * 365)
            start_date = max(earliest_start, requested_start)
        else:
            start_date = earliest_start
        
        return start_date, end_date
    
    def _parse_date(self, date_input: Union[str, datetime]) -> Optional[datetime]:
        """Parse date input to datetime object"""
        if date_input is None:
            return None
        if isinstance(date_input, datetime):
            return date_input
        if isinstance(date_input, str):
            try:
                return datetime.strptime(date_input, "%Y-%m-%d")
            except ValueError:
                logger.error(f"Invalid date format: {date_input}. Use YYYY-MM-DD")
                return None
        return None
    
    def _load_cached_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Load data from cache if available and covers the requested period"""
        cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                data_start = df.index.min()
                data_end = df.index.max()
                
                if data_start <= start_date and data_end >= end_date:
                    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
                    return filtered_df
                else:
                    logger.info(f"Cached data doesn't cover full period. Need: {start_date.date()} to {end_date.date()}, Have: {data_start.date()} to {data_end.date()}")
            except Exception as e:
                logger.error(f"Error loading cached data: {str(e)}")
        return None
    
    def _save_data(self, data: pd.DataFrame, symbol: str, timeframe: str, start_date: datetime, end_date: datetime):
        """Save data to CSV file"""
        try:
            filename = self._get_cache_filename(symbol, timeframe, start_date, end_date)
            data.to_csv(filename)
            logger.info(f"💾 Saved {len(data)} records to {filename}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def _get_cache_filename(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> str:
        """Generate cache filename for the data"""
        clean_symbol = symbol.replace(":", "_").replace("-", "_")
        timeframe_folder = os.path.join(RAW_DATA_PATH, timeframe)
        os.makedirs(timeframe_folder, exist_ok=True)
        filename = f"{clean_symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        return os.path.join(timeframe_folder, filename)