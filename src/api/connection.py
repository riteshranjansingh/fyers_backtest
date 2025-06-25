"""
Fyers API Connection Manager
Handles authentication, token management, and connection testing
"""
import os
import json
import time
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
import webbrowser
from fyers_apiv3 import fyersModel
from typing import Optional

from config.api_config import (
    FYERS_APP_ID, 
    FYERS_APP_SECRET, 
    FYERS_CLIENT_ID, 
    FYERS_REDIRECT_URI,
    CACHE_PATH
)

# Setup logging
logger = logging.getLogger(__name__)


class FyersConnection:
    """Manages Fyers API connection with token caching and auto-renewal"""
    
    def __init__(self):
        self.app_id = FYERS_APP_ID
        self.secret_key = FYERS_APP_SECRET
        self.client_id = FYERS_CLIENT_ID
        self.redirect_uri = FYERS_REDIRECT_URI
        
        # Token management
        self.token_file = os.path.join(CACHE_PATH, "fyers_token.json")
        self.access_token = None
        self.token_expiry = None
        self.fyers = None
        
    def connect(self, force_new_token: bool = False) -> bool:
        """
        Establish connection to Fyers API
        
        Args:
            force_new_token: Force new authentication even if cached token exists
            
        Returns:
            bool: True if connection successful
        """
        # Try to use cached token first
        if not force_new_token and self._load_cached_token():
            if self._create_session():
                logger.info("Connected using cached token")
                return True
                
        # Get new token
        logger.info("Getting new authentication token...")
        if self._authenticate():
            return self._create_session()
            
        return False
    
    def _authenticate(self) -> bool:
        """Handle the OAuth authentication flow"""
        try:
            # Step 1: Generate auth code URL
            session = fyersModel.SessionModel(
                client_id=self.app_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type="code",
                grant_type="authorization_code"
            )
            
            auth_url = session.generate_authcode()
            
            print("\n" + "="*50)
            print("🔐 FYERS AUTHENTICATION REQUIRED")
            print("="*50)
            print("\n1. Opening browser for authentication...")
            print(f"2. If browser doesn't open, visit: {auth_url}")
            print("3. Login and authorize the app")
            print("4. Copy the COMPLETE redirect URL from browser")
            print("\n" + "="*50 + "\n")
            
            # Open browser
            webbrowser.open(auth_url)
            
            # Get redirect URL from user
            redirect_url = input("📋 Paste the complete redirect URL here: ").strip()
            
            # Extract auth code
            parsed_url = urlparse(redirect_url)
            query_params = parse_qs(parsed_url.query)
            
            if 'auth_code' not in query_params:
                logger.error("No auth_code found in redirect URL")
                return False
                
            auth_code = query_params['auth_code'][0]
            
            # Step 2: Generate access token
            session.set_token(auth_code)
            response = session.generate_token()
            
            if 'access_token' in response:
                self.access_token = response['access_token']
                self.token_expiry = datetime.now() + timedelta(days=1)
                self._save_token()
                logger.info("✅ Authentication successful!")
                return True
            else:
                logger.error(f"Token generation failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def _create_session(self) -> bool:
        """Create Fyers API session with access token"""
        try:
            # Use the root logs directory as per our folder structure
            logs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
            os.makedirs(logs_path, exist_ok=True)
            
            self.fyers = fyersModel.FyersModel(
                client_id=self.app_id,
                token=self.access_token,
                log_path=logs_path
            )
            
            # Test the session
            profile = self.fyers.get_profile()
            if profile.get('s') == 'ok':
                logger.info(f"✅ Session created for user: {profile.get('data', {}).get('name', 'Unknown')}")
                return True
            else:
                logger.error(f"Session test failed: {profile}")
                return False
                
        except Exception as e:
            logger.error(f"Session creation error: {str(e)}")
            return False
    
    def _save_token(self):
        """Save access token to cache file"""
        token_data = {
            'access_token': self.access_token,
            'expiry': self.token_expiry.isoformat() if self.token_expiry else None,
            'saved_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
        with open(self.token_file, 'w') as f:
            json.dump(token_data, f, indent=2)
        logger.info("Token saved to cache")
    
    def _load_cached_token(self) -> bool:
        """Load token from cache if valid"""
        if not os.path.exists(self.token_file):
            return False
            
        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
                
            self.access_token = token_data.get('access_token')
            expiry_str = token_data.get('expiry')
            
            if not self.access_token or not expiry_str:
                return False
                
            self.token_expiry = datetime.fromisoformat(expiry_str)
            
            # Check if token is still valid (with 1 hour buffer)
            if datetime.now() >= self.token_expiry - timedelta(hours=1):
                logger.info("Cached token expired")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading cached token: {str(e)}")
            return False
    
    def get_session(self) -> Optional[fyersModel.FyersModel]:
        """Get active Fyers session, connecting if needed"""
        if not self.fyers:
            self.connect()
        return self.fyers
    
    def test_connection(self) -> dict:
        """Test API connection and return account info"""
        try:
            if not self.fyers:
                return {'status': 'error', 'message': 'Not connected'}
                
            profile = self.fyers.get_profile()
            if profile.get('s') == 'ok':
                data = profile.get('data', {})
                return {
                    'status': 'success',
                    'user_name': data.get('name', 'Unknown'),
                    'user_id': data.get('fy_id', 'Unknown'),
                    'email': data.get('email_id', 'Unknown'),
                    'broker': data.get('broker', 'FYERS')
                }
            else:
                return {'status': 'error', 'message': profile.get('message', 'Unknown error')}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_funds(self) -> dict:
        """Get account funds information"""
        try:
            if not self.fyers:
                return {'status': 'error', 'message': 'Not connected'}
                
            funds = self.fyers.funds()
            if funds.get('s') == 'ok':
                fund_data = funds.get('fund_limit', [{}])[0]
                return {
                    'status': 'success',
                    'balance': fund_data.get('equityAmount', 0),
                    'available': fund_data.get('equityAmount', 0) - fund_data.get('utilizedAmount', 0),
                    'utilized': fund_data.get('utilizedAmount', 0)
                }
            else:
                return {'status': 'error', 'message': funds.get('message', 'Unknown error')}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}