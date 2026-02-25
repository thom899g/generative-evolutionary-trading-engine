"""
Market Data Ingestor Module
Responsible for real-time and historical data collection from exchanges
Architecture Choice: CCXT for unified exchange interface with fallback mechanisms
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import traceback

import ccxt
import pandas as pd
import numpy as np
from firebase_admin import firestore
import yfinance as yf

# Configure logging
logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """Supported exchange types"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    FTX = "ftx"  # Note: May need alternative
    YAHOO = "yfinance"  # Fallback

@dataclass
class MarketData:
    """Structured market data container"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: Optional[int] = None
    vwap: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to Firebase-compatible dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'trades': self.trades,
            'vwap': self.vwap,
            'ingested_at': datetime.utcnow().isoformat()
        }

class MarketDataIngestor:
    """Robust market data ingestion system with fallback mechanisms"""
    
    def __init__(
        self,
        exchange_id: str = "binance",
        testnet: bool = True,
        firestore_client: Optional[firestore.Client] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize market data ingestor
        
        Args:
            exchange_id: Exchange identifier (binance, coinbase, etc.)
            testnet: Use testnet for development
            firestore_client: Firebase Firestore client for state persistence
            config: Configuration dictionary
        """
        # Validate inputs
        if not exchange_id or not isinstance(exchange_id, str):
            raise ValueError("exchange_id must be a non-empty string")
        
        self.exchange_id = exchange_id
        self.testnet = testnet
        self.firestore_client = firestore_client
        self.config = config or {}
        
        # Initialize exchange connection with error handling
        self.exchange = self._initialize_exchange()
        self.alternative_sources = self._initialize_fallbacks()
        
        # State tracking
        self._is_connected = False
        self._last_update = None
        self._error_count = 0
        self._max_errors = 10
        
        logger.info(f"MarketDataIngestor initialized for {exchange_id} (testnet: {testnet})")
    
    def _initialize_exchange