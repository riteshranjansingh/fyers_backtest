"""
Symbol Master Management
Handles fetching, storing, and searching Fyers symbols
"""
import os
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
import zipfile
import io

from config.api_config import SYMBOLS_PATH

logger = logging.getLogger(__name__)


class SymbolMaster:
    """Manages Fyers symbol master data"""
    
    def __init__(self):
        self.symbols_file = os.path.join(SYMBOLS_PATH, "symbol_master.csv")
        self.metadata_file = os.path.join(SYMBOLS_PATH, "metadata.json")
        self.symbols_df = None
        
    def fetch_symbol_master(self, exchange: str = "NSE") -> bool:
        """
        Fetch latest symbol master from Fyers
        
        Args:
            exchange: Exchange to fetch symbols for (NSE, BSE, MCX)
            
        Returns:
            bool: Success status
        """
        try:
            # Fyers symbol master URLs
            urls = {
                "NSE": "https://public.fyers.in/sym_details/NSE_CM.csv",
                "BSE": "https://public.fyers.in/sym_details/BSE_CM.csv",
                "MCX": "https://public.fyers.in/sym_details/MCX_COM.csv",
                "NSE_FO": "https://public.fyers.in/sym_details/NSE_FO.csv"
            }
            
            if exchange not in urls:
                logger.error(f"Invalid exchange: {exchange}")
                return False
            
            logger.info(f"Fetching symbol master for {exchange}...")
            
            # Download the CSV file
            response = requests.get(urls[exchange], timeout=30)
            response.raise_for_status()
            
            # Read CSV content
            df = pd.read_csv(io.StringIO(response.text))
            
            # Save to file
            os.makedirs(SYMBOLS_PATH, exist_ok=True)
            df.to_csv(self.symbols_file, index=False)
            
            # Save metadata
            metadata = {
                "last_updated": datetime.now().isoformat(),
                "exchange": exchange,
                "total_symbols": len(df),
                "columns": list(df.columns)
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.symbols_df = df
            logger.info(f"✅ Fetched {len(df)} symbols for {exchange}")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching symbol master: {str(e)}")
            return False
    
    def load_symbols(self) -> bool:
        """Load symbols from local file"""
        try:
            if os.path.exists(self.symbols_file):
                self.symbols_df = pd.read_csv(self.symbols_file)
                logger.info(f"Loaded {len(self.symbols_df)} symbols from cache")
                return True
            else:
                logger.warning("No cached symbols found")
                return False
        except Exception as e:
            logger.error(f"Error loading symbols: {str(e)}")
            return False
    
    def should_update(self, days: int = 14) -> bool:
        """Check if symbol master needs update"""
        if not os.path.exists(self.metadata_file):
            return True
            
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                
            last_updated = datetime.fromisoformat(metadata['last_updated'])
            return datetime.now() - last_updated > timedelta(days=days)
            
        except Exception:
            return True
    
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for symbols by name or symbol code
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching symbols
        """
        if self.symbols_df is None:
            self.load_symbols()
            
        if self.symbols_df is None or self.symbols_df.empty:
            return []
            
        query = query.upper()
        results = []
        
        # Common column mappings (adjust based on actual CSV structure)
        symbol_col = '13'  # Fyers symbol column
        name_col = '2'     # Company name column
        
        # Search in symbol and name
        mask = (
            self.symbols_df[symbol_col].str.contains(query, case=False, na=False) |
            self.symbols_df[name_col].str.contains(query, case=False, na=False)
        )
        
        matches = self.symbols_df[mask].head(limit)
        
        for _, row in matches.iterrows():
            results.append({
                'symbol': row[symbol_col],
                'name': row[name_col],
                'exchange': 'NSE',  # Add exchange info
                'type': 'EQ'  # Add instrument type
            })
            
        return results
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed information for a specific symbol"""
        if self.symbols_df is None:
            self.load_symbols()
            
        if self.symbols_df is None:
            return None
            
        symbol_col = '13'
        mask = self.symbols_df[symbol_col] == symbol
        
        if mask.any():
            row = self.symbols_df[mask].iloc[0]
            return {
                'symbol': row[symbol_col],
                'name': row['2'],
                'isin': row.get('12', ''),
                'lot_size': row.get('14', 1)
            }
        
        return None
    
    def get_popular_symbols(self) -> List[Dict]:
        """Get list of popular symbols for quick access"""
        popular = [
            {'symbol': 'NSE:NIFTY50-INDEX', 'name': 'Nifty 50 Index', 'type': 'INDEX'},
            {'symbol': 'NSE:NIFTYBANK-INDEX', 'name': 'Nifty Bank Index', 'type': 'INDEX'},
            {'symbol': 'NSE:RELIANCE-EQ', 'name': 'Reliance Industries', 'type': 'EQ'},
            {'symbol': 'NSE:TCS-EQ', 'name': 'Tata Consultancy Services', 'type': 'EQ'},
            {'symbol': 'NSE:INFY-EQ', 'name': 'Infosys', 'type': 'EQ'},
            {'symbol': 'NSE:HDFC-EQ', 'name': 'HDFC Ltd', 'type': 'EQ'},
            {'symbol': 'NSE:ICICIBANK-EQ', 'name': 'ICICI Bank', 'type': 'EQ'},
            {'symbol': 'NSE:SBIN-EQ', 'name': 'State Bank of India', 'type': 'EQ'},
            {'symbol': 'NSE:BHARTIARTL-EQ', 'name': 'Bharti Airtel', 'type': 'EQ'},
            {'symbol': 'NSE:ITC-EQ', 'name': 'ITC Limited', 'type': 'EQ'}
        ]
        return popular
    
    def validate_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid"""
        # Basic format validation
        if ':' not in symbol:
            return False
            
        # Check if symbol exists in master
        if self.symbols_df is not None:
            symbol_col = '13'
            return symbol in self.symbols_df[symbol_col].values
            
        # If no master loaded, do basic validation
        parts = symbol.split(':')
        return len(parts) == 2 and parts[0] in ['NSE', 'BSE', 'MCX']