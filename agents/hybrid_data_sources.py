"""
Hybrid Data Sources for RL Pre-training
Combines historical market data, simulated regimes, and LLM-fabricated sentiment
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import random

# Handle optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from tools.alpaca_market_data import fetch_stock_history

logger = logging.getLogger(__name__)

@dataclass
class MarketRegimeData:
    """Market regime characteristics for simulation."""
    regime_type: str  # bull, bear, sideways, volatile, crisis
    volatility: float  # Daily volatility 0.01-0.10
    trend_strength: float  # -1.0 to 1.0
    mean_reversion: float  # 0.0 to 1.0
    correlation_decay: float  # How quickly correlations break down
    event_frequency: float  # Frequency of shock events
    duration_days: int  # Typical regime duration

@dataclass
class SentimentData:
    """LLM-fabricated sentiment data."""
    overall_sentiment: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    news_sentiment: float
    social_sentiment: float
    earnings_sentiment: float
    analyst_sentiment: float
    volatility_sentiment: float  # Expected volatility based on sentiment
    coherence_score: float  # How coherent the sentiment signals are

@dataclass
class L2OrderBookData:
    """Level 2 order book data."""
    bid_prices: List[float]
    bid_sizes: List[float]
    ask_prices: List[float]
    ask_sizes: List[float]
    spread: float
    depth: float
    imbalance: float  # Order flow imbalance

class MarketRegimeGenerator:
    """Generates realistic market regimes with proper transitions."""
    
    def __init__(self):
        self.regime_templates = {
            'bull': MarketRegimeData('bull', 0.015, 0.7, 0.3, 0.8, 0.1, 120),
            'bear': MarketRegimeData('bear', 0.025, -0.6, 0.4, 0.6, 0.2, 90),
            'sideways': MarketRegimeData('sideways', 0.012, 0.1, 0.8, 0.9, 0.05, 180),
            'volatile': MarketRegimeData('volatile', 0.035, 0.2, 0.2, 0.4, 0.4, 45),
            'crisis': MarketRegimeData('crisis', 0.060, -0.8, 0.1, 0.2, 0.8, 30)
        }
        
        # Transition probabilities between regimes
        self.transition_matrix = {
            'bull': {'bull': 0.85, 'sideways': 0.10, 'volatile': 0.04, 'bear': 0.01},
            'bear': {'bear': 0.80, 'crisis': 0.05, 'volatile': 0.10, 'sideways': 0.05},
            'sideways': {'sideways': 0.75, 'bull': 0.15, 'bear': 0.05, 'volatile': 0.05},
            'volatile': {'volatile': 0.60, 'bull': 0.20, 'bear': 0.10, 'crisis': 0.05, 'sideways': 0.05},
            'crisis': {'crisis': 0.70, 'bear': 0.20, 'volatile': 0.10}
        }
    
    def generate_regime_sequence(self, days: int, start_regime: str = 'bull') -> List[MarketRegimeData]:
        """Generate a sequence of market regimes."""
        regimes = []
        current_regime = start_regime
        days_in_regime = 0
        
        for day in range(days):
            regime_template = self.regime_templates[current_regime]
            
            # Add some noise to the regime parameters
            regime = MarketRegimeData(
                regime_type=regime_template.regime_type,
                volatility=max(0.005, regime_template.volatility * np.random.uniform(0.7, 1.3)),
                trend_strength=regime_template.trend_strength * np.random.uniform(0.8, 1.2),
                mean_reversion=np.clip(regime_template.mean_reversion * np.random.uniform(0.9, 1.1), 0, 1),
                correlation_decay=np.clip(regime_template.correlation_decay * np.random.uniform(0.9, 1.1), 0, 1),
                event_frequency=regime_template.event_frequency * np.random.uniform(0.5, 2.0),
                duration_days=regime_template.duration_days
            )
            
            regimes.append(regime)
            days_in_regime += 1
            
            # Check for regime transition
            if days_in_regime > regime_template.duration_days * np.random.uniform(0.5, 1.5):
                # Transition to new regime
                transitions = self.transition_matrix.get(current_regime, {})
                if transitions:
                    new_regime = np.random.choice(
                        list(transitions.keys()), 
                        p=list(transitions.values())
                    )
                    current_regime = new_regime
                    days_in_regime = 0
        
        return regimes

class LLMSentimentFabricator:
    """Fabricates realistic sentiment data using patterns and relationships."""
    
    def __init__(self):
        self.sentiment_patterns = {
            'bull': {'base': 0.6, 'noise': 0.2, 'trend': 0.05},
            'bear': {'base': -0.5, 'noise': 0.3, 'trend': -0.03},
            'sideways': {'base': 0.1, 'noise': 0.4, 'trend': 0.0},
            'volatile': {'base': 0.0, 'noise': 0.6, 'trend': 0.0},
            'crisis': {'base': -0.8, 'noise': 0.4, 'trend': -0.1}
        }
        
        # Correlation between different sentiment sources
        self.source_correlations = np.array([
            [1.0, 0.7, 0.5, 0.6, 0.3],  # news
            [0.7, 1.0, 0.8, 0.4, 0.2],  # social
            [0.5, 0.8, 1.0, 0.3, 0.1],  # earnings
            [0.6, 0.4, 0.3, 1.0, 0.4],  # analyst
            [0.3, 0.2, 0.1, 0.4, 1.0]   # volatility
        ])
        
        self.previous_sentiment = None
        
    def generate_sentiment_data(self, regime: MarketRegimeData, price_return: float = 0.0) -> SentimentData:
        """Generate realistic sentiment data based on market regime and price action."""
        
        pattern = self.sentiment_patterns[regime.regime_type]
        
        # Base sentiment influenced by regime and recent price action
        base_sentiment = pattern['base'] + price_return * 10  # Price influence
        
        # Generate correlated sentiment sources
        if self.previous_sentiment is None:
            # First generation - use base patterns
            raw_sentiments = np.random.multivariate_normal(
                mean=[base_sentiment] * 5,
                cov=self.source_correlations * pattern['noise']
            )
        else:
            # Use momentum from previous sentiment
            momentum = 0.7  # Sentiment persistence
            innovation = np.random.multivariate_normal(
                mean=[base_sentiment] * 5,
                cov=self.source_correlations * pattern['noise']
            )
            raw_sentiments = (momentum * np.array([
                self.previous_sentiment.news_sentiment,
                self.previous_sentiment.social_sentiment,
                self.previous_sentiment.earnings_sentiment,
                self.previous_sentiment.analyst_sentiment,
                self.previous_sentiment.volatility_sentiment
            ]) + (1 - momentum) * innovation)
        
        # Clip to valid range
        sentiments = np.clip(raw_sentiments, -1.0, 1.0)
        
        # Calculate overall sentiment as weighted average
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # news, social, earnings, analyst, volatility
        overall_sentiment = np.dot(sentiments, weights)
        
        # Calculate confidence based on coherence
        coherence_score = 1.0 - np.std(sentiments) / 2.0  # Higher std = lower coherence
        confidence = np.clip(coherence_score * np.random.uniform(0.8, 1.2), 0.3, 1.0)
        
        # Calculate volatility sentiment (independent but correlated with overall)
        vol_sentiment = sentiments[4]
        
        sentiment_data = SentimentData(
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            news_sentiment=sentiments[0],
            social_sentiment=sentiments[1],
            earnings_sentiment=sentiments[2],
            analyst_sentiment=sentiments[3],
            volatility_sentiment=vol_sentiment,
            coherence_score=coherence_score
        )
        
        self.previous_sentiment = sentiment_data
        return sentiment_data

class L2OrderBookSimulator:
    """Simulates realistic Level 2 order book data."""
    
    def __init__(self):
        self.book_depth = 10  # Number of levels
        self.tick_size = 0.01
        
    def generate_order_book(self, 
                          mid_price: float, 
                          regime: MarketRegimeData,
                          sentiment: SentimentData) -> L2OrderBookData:
        """Generate realistic L2 order book data."""
        
        # Base spread influenced by volatility and sentiment uncertainty
        base_spread = mid_price * (0.0001 + regime.volatility * 0.01)
        sentiment_spread_factor = 1.0 + (1.0 - sentiment.confidence) * 0.5
        spread = base_spread * sentiment_spread_factor
        
        # Generate bid/ask levels
        bid_prices = []
        ask_prices = []
        bid_sizes = []
        ask_sizes = []
        
        # Sentiment bias affects order book shape
        sentiment_bias = sentiment.overall_sentiment * 0.3
        
        for level in range(self.book_depth):
            # Exponential decay of size with distance from mid
            size_decay = np.exp(-level * 0.5)
            
            # Base size influenced by volatility (higher vol = more liquidity needed)
            base_size = np.random.uniform(100, 1000) * size_decay * (1 + regime.volatility * 2)
            
            # Bid side (buy orders)
            bid_price = mid_price - spread/2 - level * self.tick_size
            bid_size = base_size * (1 + sentiment_bias)  # More bids if positive sentiment
            
            # Ask side (sell orders) 
            ask_price = mid_price + spread/2 + level * self.tick_size
            ask_size = base_size * (1 - sentiment_bias)  # Fewer asks if positive sentiment
            
            bid_prices.append(bid_price)
            ask_prices.append(ask_price)
            bid_sizes.append(max(10, bid_size))  # Minimum size
            ask_sizes.append(max(10, ask_size))
        
        # Calculate order flow imbalance
        total_bid_size = sum(bid_sizes[:3])  # Top 3 levels
        total_ask_size = sum(ask_sizes[:3])
        imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        
        # Calculate depth as total size at top levels
        depth = (total_bid_size + total_ask_size) / 2
        
        return L2OrderBookData(
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
            spread=spread,
            depth=depth,
            imbalance=imbalance
        )

class HistoricalDataEnhancer:
    """Enhances historical market data with additional features."""
    
    def __init__(self):
        self.cache_dir = Path("data/historical_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    async def fetch_enhanced_historical_data(self, 
                                           symbols: List[str], 
                                           period: str = "2y",
                                           interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch and enhance historical data for multiple symbols."""
        
        enhanced_data = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_file = self.cache_dir / f"{symbol}_{period}_{interval}.parquet"
                
                if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 1:
                    df = pd.read_parquet(cache_file)
                    logger.debug(f"Loaded {symbol} from cache")
                else:
                    # Fetch fresh data using utility function
                    df = fetch_stock_history(symbol, period=period, interval=interval)
                    
                    if df is None or len(df) == 0:
                        logger.warning(f"No data retrieved for {symbol}")
                        continue
                    
                    # Enhance with technical indicators
                    df = self._add_technical_indicators(df)
                    
                    # Cache the enhanced data
                    df.to_parquet(cache_file)
                    logger.debug(f"Cached enhanced data for {symbol}")
                
                enhanced_data[symbol] = df
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
        
        return enhanced_data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to OHLCV data."""
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility measures
        df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['parkinson_vol'] = np.sqrt(
            252 * (np.log(df['High'] / df['Low']) ** 2) / (4 * np.log(2))
        ).rolling(20).mean()
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Price position indicators
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return df

class HybridDataGenerator:
    """Main class that combines all data sources for RL pre-training."""
    
    def __init__(self):
        self.regime_generator = MarketRegimeGenerator()
        self.sentiment_fabricator = LLMSentimentFabricator()
        self.orderbook_simulator = L2OrderBookSimulator()
        self.historical_enhancer = HistoricalDataEnhancer()
        
    async def generate_training_dataset(self,
                                      symbols: List[str],
                                      training_days: int = 1000,
                                      validation_split: float = 0.2) -> Dict[str, Any]:
        """Generate comprehensive training dataset combining all sources."""
        
        logger.info(f"Generating hybrid training dataset for {len(symbols)} symbols over {training_days} days")
        
        # 1. Fetch historical data
        historical_data = await self.historical_enhancer.fetch_enhanced_historical_data(symbols, period="2y")
        
        # 2. Generate market regime sequence
        regimes = self.regime_generator.generate_regime_sequence(training_days)
        
        # 3. Generate combined dataset
        dataset = {
            'market_data': [],
            'sentiment_data': [],
            'orderbook_data': [],
            'regime_data': [],
            'metadata': {
                'symbols': symbols,
                'training_days': training_days,
                'generation_time': datetime.now().isoformat(),
                'validation_split': validation_split
            }
        }
        
        for day_idx, regime in enumerate(regimes):
            daily_data = {}
            
            for symbol in symbols:
                # Get historical baseline if available
                if symbol in historical_data and len(historical_data[symbol]) > day_idx:
                    hist_row = historical_data[symbol].iloc[day_idx % len(historical_data[symbol])]
                    base_price = hist_row['Close']
                    base_volume = hist_row['Volume']
                    base_return = hist_row.get('returns', 0.0)
                else:
                    # Use synthetic baseline
                    base_price = 100 * np.random.uniform(0.5, 2.0)
                    base_volume = 1000000 * np.random.uniform(0.1, 3.0)
                    base_return = np.random.normal(0, regime.volatility)
                
                # Generate synthetic price using regime characteristics
                if day_idx == 0:
                    price = base_price
                else:
                    prev_price = daily_data.get(symbol, {}).get('price', base_price)
                    
                    # Trend component
                    trend = regime.trend_strength * 0.001
                    
                    # Mean reversion component
                    deviation = (prev_price - base_price) / base_price
                    mean_reversion = -regime.mean_reversion * deviation * 0.01
                    
                    # Random shock
                    shock = np.random.normal(0, regime.volatility)
                    
                    # Event shocks
                    if np.random.random() < regime.event_frequency:
                        event_shock = np.random.normal(0, regime.volatility * 3) * np.random.choice([-1, 1])
                        shock += event_shock
                    
                    total_return = trend + mean_reversion + shock
                    price = prev_price * (1 + total_return)
                
                # Generate sentiment data
                sentiment = self.sentiment_fabricator.generate_sentiment_data(regime, total_return if day_idx > 0 else 0)
                
                # Generate order book data
                orderbook = self.orderbook_simulator.generate_order_book(price, regime, sentiment)
                
                daily_data[symbol] = {
                    'price': price,
                    'volume': base_volume * (1 + np.random.uniform(-0.2, 0.2)),
                    'return': total_return if day_idx > 0 else 0,
                    'sentiment': asdict(sentiment),
                    'orderbook': asdict(orderbook)
                }
            
            dataset['market_data'].append(daily_data)
            dataset['regime_data'].append(asdict(regime))
            
            if day_idx % 100 == 0:
                logger.info(f"Generated {day_idx}/{training_days} days of training data")
        
        # Split into training and validation
        split_idx = int(training_days * (1 - validation_split))
        
        training_data = {
            'market_data': dataset['market_data'][:split_idx],
            'regime_data': dataset['regime_data'][:split_idx],
            'metadata': dataset['metadata'].copy()
        }
        training_data['metadata']['split'] = 'training'
        training_data['metadata']['days'] = split_idx
        
        validation_data = {
            'market_data': dataset['market_data'][split_idx:],
            'regime_data': dataset['regime_data'][split_idx:],
            'metadata': dataset['metadata'].copy()
        }
        validation_data['metadata']['split'] = 'validation'
        validation_data['metadata']['days'] = training_days - split_idx
        
        logger.info(f"✅ Generated {training_days} days of hybrid training data")
        logger.info(f"   Training: {split_idx} days, Validation: {training_days - split_idx} days")
        logger.info(f"   Regimes: {len(set(r['regime_type'] for r in dataset['regime_data']))} unique types")
        
        return {
            'training': training_data,
            'validation': validation_data,
            'full_dataset': dataset
        }
    
    def save_dataset(self, dataset: Dict[str, Any], filepath: str):
        """Save dataset to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        logger.info(f"💾 Saved dataset to {filepath}")
    
    def load_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load dataset from disk."""
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        
        logger.info(f"📂 Loaded dataset from {filepath}")
        return dataset

# Utility functions for RL integration
def convert_to_rl_observations(market_data: List[Dict], lookback_window: int = 10) -> np.ndarray:
    """Convert market data to RL observation format."""
    
    observations = []
    
    for i in range(lookback_window, len(market_data)):
        obs_window = []
        
        for j in range(i - lookback_window, i):
            day_data = market_data[j]
            
            # Flatten all symbol data for this day
            day_features = []
            for symbol, data in day_data.items():
                day_features.extend([
                    data['price'],
                    data['volume'],
                    data['return'],
                    data['sentiment']['overall_sentiment'],
                    data['sentiment']['confidence'],
                    data['orderbook']['spread'],
                    data['orderbook']['imbalance']
                ])
            
            obs_window.append(day_features)
        
        observations.append(obs_window)
    
    return np.array(observations)

if __name__ == "__main__":
    # Test the hybrid data generation
    async def test_hybrid_data_generation():
        generator = HybridDataGenerator()
        
        # Generate small test dataset
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        dataset = await generator.generate_training_dataset(
            symbols=symbols,
            training_days=100,
            validation_split=0.2
        )
        
        # Save test dataset
        generator.save_dataset(dataset, 'data/test_hybrid_dataset.json')
        
        # Test RL observation conversion
        training_data = dataset['training']['market_data']
        observations = convert_to_rl_observations(training_data, lookback_window=5)
        
        print(f"Generated dataset with {len(training_data)} training days")
        print(f"RL observations shape: {observations.shape}")
        print(f"Sample regime types: {set(r['regime_type'] for r in dataset['training']['regime_data'])}")
    
    asyncio.run(test_hybrid_data_generation())