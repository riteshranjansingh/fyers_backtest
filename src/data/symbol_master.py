"""
Symbol Master Management - CLEAN VERSION
Handles fetching, storing, and searching Fyers symbols with correct column mappings
"""
import os
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
import io

from config.api_config import SYMBOLS_PATH

logger = logging.getLogger(__name__)


class SymbolMaster:
    """Manages Fyers symbol master data with corrected column mappings"""
    
    def __init__(self):
        self.symbols_file = os.path.join(SYMBOLS_PATH, "symbol_master.csv")
        self.metadata_file = os.path.join(SYMBOLS_PATH, "metadata.json")
        self.symbols_df = None
        self.column_map = None
        
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
            
            # Read CSV content and inspect structure
            df = pd.read_csv(io.StringIO(response.text))
            
            # Auto-detect column mappings by examining the data
            self._detect_column_mappings(df)
            
            # Save to file
            os.makedirs(SYMBOLS_PATH, exist_ok=True)
            df.to_csv(self.symbols_file, index=False)
            
            # Save metadata with column mappings
            metadata = {
                "last_updated": datetime.now().isoformat(),
                "exchange": exchange,
                "total_symbols": len(df),
                "columns": list(df.columns),
                "column_mappings": self.column_map
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.symbols_df = df
            logger.info(f"✅ Fetched {len(df)} symbols for {exchange}")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching symbol master: {str(e)}")
            return False
    
    def _detect_column_mappings(self, df: pd.DataFrame):
        """Auto-detect which columns contain symbol and name data based on actual CSV structure"""
        self.column_map = {
            'symbol': None,
            'name': None,
            'isin': None,
            'short_symbol': None
        }
        
        # Based on your CSV structure analysis:
        # Column 1: Company names
        # Column 9: Fyers symbols (NSE:SYMBOL-EQ format)
        # Column 13: Short symbols
        # Column 5: ISIN codes
        
        # Check if we have the expected structure
        try:
            # Column 9 should contain NSE: format symbols
            if len(df.columns) > 9:
                col_9_sample = df.iloc[:5, 9].astype(str)
                if col_9_sample.str.contains('NSE:|BSE:|MCX:', case=False, na=False).any():
                    self.column_map['symbol'] = df.columns[9]
                    logger.info(f"Found symbol column at index 9: {df.columns[9]}")
            
            # Column 1 should contain company names
            if len(df.columns) > 1:
                col_1_sample = df.iloc[:5, 1].astype(str)
                if col_1_sample.str.contains('LTD|LIMITED|BANK|INDUSTRIES|COMPANY', case=False, na=False).any():
                    self.column_map['name'] = df.columns[1]
                    logger.info(f"Found name column at index 1: {df.columns[1]}")
            
            # Column 5 should contain ISIN codes
            if len(df.columns) > 5:
                col_5_sample = df.iloc[:5, 5].astype(str)
                if col_5_sample.str.match(r'^IN[A-Z0-9]{10}$', na=False).any():
                    self.column_map['isin'] = df.columns[5]
                    logger.info(f"Found ISIN column at index 5: {df.columns[5]}")
            
            # Column 13 should contain short symbols
            if len(df.columns) > 13:
                self.column_map['short_symbol'] = df.columns[13]
                logger.info(f"Found short symbol column at index 13: {df.columns[13]}")
            
        except Exception as e:
            logger.error(f"Error in specific column detection: {e}")
            # Fallback to general detection
            self._fallback_column_detection(df)
        
        logger.info(f"Final column mappings: {self.column_map}")
    
    def _fallback_column_detection(self, df: pd.DataFrame):
        """Fallback method to detect columns if standard structure doesn't match"""
        # Look for symbol column (contains exchange prefix like NSE:)
        for i, col in enumerate(df.columns):
            sample_values = df[col].astype(str).head(10)
            if sample_values.str.contains('NSE:|BSE:|MCX:', case=False, na=False).any():
                self.column_map['symbol'] = col
                logger.info(f"Fallback: Found symbol column at {i}: {col}")
                break
        
        # Look for company name column (contains company names)
        for i, col in enumerate(df.columns):
            if col != self.column_map['symbol']:
                sample_values = df[col].astype(str).head(10)
                if sample_values.str.contains('LTD|LIMITED|BANK|INDUSTRIES|COMPANY', case=False, na=False).any():
                    self.column_map['name'] = col
                    logger.info(f"Fallback: Found name column at {i}: {col}")
                    break
        
        # Look for ISIN column (12-character alphanumeric starting with IN)
        for i, col in enumerate(df.columns):
            sample_values = df[col].astype(str).head(10)
            if sample_values.str.match(r'^IN[A-Z0-9]{10}$', na=False).any():
                self.column_map['isin'] = col
                logger.info(f"Fallback: Found ISIN column at {i}: {col}")
                break
    
    def _update_metadata(self):
        """Update metadata file with current column mappings"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            metadata['column_mappings'] = self.column_map
            metadata['mappings_updated'] = datetime.now().isoformat()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info("Metadata updated with column mappings")
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
    def load_symbols(self) -> bool:
        """Load symbols from local file"""
        try:
            if os.path.exists(self.symbols_file):
                self.symbols_df = pd.read_csv(self.symbols_file)
                
                # Try to load column mappings from metadata
                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.column_map = metadata.get('column_mappings', {})
                
                # If no column mappings, detect them now
                if not self.column_map or not all(self.column_map.values()):
                    logger.info("Column mappings missing, detecting now...")
                    self._detect_column_mappings(self.symbols_df)
                    
                    # Save updated metadata
                    self._update_metadata()
                
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
            
        if self.symbols_df is None or self.symbols_df.empty or not self.column_map:
            logger.error("No data loaded or column mappings missing")
            return []
            
        query = query.upper()
        results = []
        
        symbol_col = self.column_map.get('symbol')
        name_col = self.column_map.get('name')
        short_symbol_col = self.column_map.get('short_symbol')
        
        if not symbol_col or not name_col:
            logger.error(f"Required column mappings not found: symbol={symbol_col}, name={name_col}")
            return []
        
        # Search in symbol, name, and short symbol
        try:
            # Create search masks
            symbol_mask = self.symbols_df[symbol_col].astype(str).str.contains(query, case=False, na=False)
            name_mask = self.symbols_df[name_col].astype(str).str.contains(query, case=False, na=False)
            
            # Also search in short symbol if available
            if short_symbol_col:
                short_mask = self.symbols_df[short_symbol_col].astype(str).str.contains(query, case=False, na=False)
                combined_mask = symbol_mask | name_mask | short_mask
            else:
                combined_mask = symbol_mask | name_mask
            
            matches = self.symbols_df[combined_mask].head(limit)
            
            for _, row in matches.iterrows():
                # Extract exchange and type from symbol
                full_symbol = str(row[symbol_col])
                if ':' in full_symbol:
                    exchange, symbol_part = full_symbol.split(':', 1)
                    instrument_type = 'EQ' if '-EQ' in symbol_part else 'INDEX' if 'INDEX' in symbol_part else 'OTHER'
                else:
                    exchange = 'NSE'
                    instrument_type = 'EQ'
                
                results.append({
                    'symbol': full_symbol,
                    'name': str(row[name_col]),
                    'exchange': exchange,
                    'type': instrument_type
                })
                
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            logger.error(f"Available columns: {list(self.symbols_df.columns)}")
            logger.error(f"Column mappings: {self.column_map}")
            
        return results
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed information for a specific symbol"""
        if self.symbols_df is None:
            self.load_symbols()
            
        if self.symbols_df is None or not self.column_map:
            return None
            
        symbol_col = self.column_map.get('symbol')
        name_col = self.column_map.get('name')
        isin_col = self.column_map.get('isin')
        
        if not symbol_col:
            logger.error("Symbol column mapping not found")
            return None
            
        try:
            mask = self.symbols_df[symbol_col].astype(str) == symbol
            
            if mask.any():
                row = self.symbols_df[mask].iloc[0]
                return {
                    'symbol': str(row[symbol_col]),
                    'name': str(row[name_col]) if name_col else 'Unknown',
                    'isin': str(row[isin_col]) if isin_col and pd.notna(row[isin_col]) else '',
                    'lot_size': 1  # Default lot size
                }
        except Exception as e:
            logger.error(f"Symbol info error: {str(e)}")
        
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
        if self.symbols_df is not None and self.column_map:
            symbol_col = self.column_map.get('symbol')
            if symbol_col:
                try:
                    return symbol in self.symbols_df[symbol_col].astype(str).values
                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    
        # If no master loaded, do basic validation
        parts = symbol.split(':')
        return len(parts) == 2 and parts[0] in ['NSE', 'BSE', 'MCX']
    
    def get_debug_info(self) -> Dict:
        """Get debug information about loaded data"""
        if self.symbols_df is None:
            return {"status": "No data loaded"}
            
        return {
            "total_symbols": len(self.symbols_df),
            "columns": list(self.symbols_df.columns),
            "column_mappings": self.column_map,
            "sample_data": self.symbols_df.head(3).to_dict('records')
        }