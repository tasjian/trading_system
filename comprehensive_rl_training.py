#!/usr/bin/env python3
"""
Comprehensive RL Training Script
Runs deep training for the RL agent using Alpaca market data and yfinance data
with improved episode length, reward engineering, and state representation.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import yfinance as yf
import time

# Import our systems
from agents.comprehensive_rl_pretraining import create_comprehensive_rl_system, OnlineLearningConfig, SafetyConstraints
from agents.realistic_trading_env import RealisticTradingEnvironment
from tools.alpaca_client import alpaca_client
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveRLTrainer:
    """Comprehensive RL training with enhanced data sources and longer episodes."""
    
    def __init__(self):
        """Initialize the comprehensive RL trainer."""
        self.symbols = self._get_training_symbols()
        self.rl_system = None
        self.training_data = {}
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'avg_rewards': [],
            'best_episode_reward': -float('inf'),
            'training_start': None,
            'data_sources': []
        }
        
    def _get_training_symbols(self) -> List[str]:
        """Get a diverse set of symbols for training."""
        
        # Use a mix of different asset classes and sectors for robust training
        training_symbols = [
            # Large cap tech (high volatility)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
            
            # Financial sector
            'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA',
            
            # Healthcare & consumer goods
            'JNJ', 'PG', 'KO', 'PFE', 'UNH', 'WMT',
            
            # Industrial & energy 
            'CAT', 'BA', 'XOM', 'CVX', 'NEE',
            
            # ETFs for diversification
            'SPY', 'QQQ', 'XLF', 'XLE', 'XLK'
        ]
        
        return training_symbols[:10]  # FIXED: Use exactly 10 symbols to match RL dimensions
    
    async def prepare_comprehensive_training_data(self) -> Dict[str, Any]:
        """Prepare comprehensive training data from multiple sources."""
        
        logger.info("🔄 Preparing comprehensive training data...")
        
        training_data = {
            'alpaca_data': {},
            'yfinance_data': {},
            'combined_features': {},
            'metadata': {
                'symbols': self.symbols,
                'date_range': None,
                'total_bars': 0,
                'data_sources': []
            }
        }
        
        # Define training period (1 year of data)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        training_data['metadata']['date_range'] = {
            'start': start_date.isoformat(),
            'end': end_date.isoformat()
        }
        
        # 1. Get Alpaca data
        logger.info("📊 Fetching Alpaca market data...")
        alpaca_data = await self._fetch_alpaca_data(start_date, end_date)
        training_data['alpaca_data'] = alpaca_data
        training_data['metadata']['data_sources'].append('alpaca')
        
        # 2. Get yfinance data (more comprehensive)
        logger.info("📈 Fetching yfinance market data...")
        yfinance_data = await self._fetch_yfinance_data(start_date, end_date)
        training_data['yfinance_data'] = yfinance_data
        training_data['metadata']['data_sources'].append('yfinance')
        
        # 3. Combine and enhance features
        logger.info("🔧 Creating enhanced feature sets...")
        combined_features = self._create_enhanced_features(alpaca_data, yfinance_data)
        training_data['combined_features'] = combined_features
        
        # 4. Calculate statistics
        total_bars = sum(len(data) for data in combined_features.values() if isinstance(data, list))
        training_data['metadata']['total_bars'] = total_bars
        
        logger.info(f"✅ Training data prepared: {len(self.symbols)} symbols, {total_bars} total bars")
        
        return training_data
    
    async def _fetch_alpaca_data(self, start_date: datetime, end_date: datetime) -> Dict[str, List]:
        """Fetch historical data from Alpaca."""
        
        alpaca_data = {}
        
        for symbol in self.symbols:
            try:
                # Get 1-day bars from Alpaca
                df = alpaca_client.get_market_data(symbol, timeframe="1Day")
                
                if df is not None and len(df) > 0:
                    # Convert to list format for consistent processing
                    alpaca_data[symbol] = {
                        'timestamp': df.index.tolist(),
                        'open': df['open'].tolist(),
                        'high': df['high'].tolist(), 
                        'low': df['low'].tolist(),
                        'close': df['close'].tolist(),
                        'volume': df['volume'].tolist()
                    }
                    
                    logger.debug(f"✅ Alpaca: {symbol} - {len(df)} bars")
                else:
                    logger.warning(f"❌ Alpaca: No data for {symbol}")
                    
            except Exception as e:
                logger.warning(f"❌ Alpaca error for {symbol}: {e}")
                
        return alpaca_data
    
    async def _fetch_yfinance_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Fetch historical data from yfinance."""
        
        yfinance_data = {}
        
        # Batch download for efficiency
        try:
            tickers_str = ' '.join(self.symbols)
            data = yf.download(
                tickers_str,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            if len(self.symbols) == 1:
                # Single ticker case
                symbol = self.symbols[0]
                if not data.empty:
                    yfinance_data[symbol] = {
                        'timestamp': data.index.tolist(),
                        'open': data['Open'].ffill().tolist(),
                        'high': data['High'].ffill().tolist(),
                        'low': data['Low'].ffill().tolist(),
                        'close': data['Close'].ffill().tolist(),
                        'volume': data['Volume'].fillna(0).tolist()
                    }
                    logger.debug(f"✅ yfinance: {symbol} - {len(data)} bars")
            else:
                # Multiple tickers case
                for symbol in self.symbols:
                    try:
                        if symbol in data.columns.levels[0]:
                            symbol_data = data[symbol].dropna()
                            if not symbol_data.empty:
                                yfinance_data[symbol] = {
                                    'timestamp': symbol_data.index.tolist(),
                                    'open': symbol_data['Open'].ffill().tolist(),
                                    'high': symbol_data['High'].ffill().tolist(),
                                    'low': symbol_data['Low'].ffill().tolist(),
                                    'close': symbol_data['Close'].ffill().tolist(),
                                    'volume': symbol_data['Volume'].fillna(0).tolist()
                                }
                                logger.debug(f"✅ yfinance: {symbol} - {len(symbol_data)} bars")
                            else:
                                logger.warning(f"❌ yfinance: Empty data for {symbol}")
                        else:
                            logger.warning(f"❌ yfinance: {symbol} not in downloaded data")
                    except Exception as e:
                        logger.warning(f"❌ yfinance error for {symbol}: {e}")
                        
        except Exception as e:
            logger.error(f"❌ yfinance batch download failed: {e}")
            
        return yfinance_data
    
    def _create_enhanced_features(self, alpaca_data: Dict, yfinance_data: Dict) -> Dict[str, Any]:
        """Create enhanced feature sets combining Alpaca and yfinance data."""
        
        enhanced_features = {}
        
        for symbol in self.symbols:
            try:
                # Prefer yfinance data (more comprehensive), fallback to Alpaca
                primary_data = yfinance_data.get(symbol, alpaca_data.get(symbol))
                secondary_data = alpaca_data.get(symbol, yfinance_data.get(symbol))
                
                if not primary_data:
                    logger.warning(f"❌ No data available for {symbol}")
                    continue
                
                # Convert to pandas for feature engineering
                df = pd.DataFrame({
                    'timestamp': primary_data['timestamp'],
                    'open': primary_data['open'],
                    'high': primary_data['high'],
                    'low': primary_data['low'],
                    'close': primary_data['close'],
                    'volume': primary_data['volume']
                })
                
                # Set timestamp as index
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                # Technical indicators
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                df['volatility'] = df['returns'].rolling(20).std()
                
                # Price-based features
                df['sma_20'] = df['close'].rolling(20).mean()
                df['sma_50'] = df['close'].rolling(50).mean()
                df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
                df['price_above_sma50'] = (df['close'] > df['sma_50']).astype(int)
                
                # Momentum features
                df['rsi'] = self._calculate_rsi(df['close'])
                df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
                df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
                
                # Volume features
                df['volume_ma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
                df['price_volume'] = df['close'] * df['volume']
                
                # Volatility regime
                df['vol_regime'] = (df['volatility'] > df['volatility'].rolling(60).quantile(0.7)).astype(int)
                
                # Trend features
                df['trend_5'] = (df['close'] > df['close'].shift(5)).astype(int)
                df['trend_20'] = (df['close'] > df['close'].shift(20)).astype(int)
                
                # Clean data
                df = df.ffill().fillna(0)
                
                # Store enhanced features
                enhanced_features[symbol] = df
                
                logger.debug(f"✅ Enhanced features for {symbol}: {len(df)} bars, {len(df.columns)} features")
                
            except Exception as e:
                logger.error(f"❌ Feature engineering failed for {symbol}: {e}")
                
        logger.info(f"✅ Enhanced features created for {len(enhanced_features)} symbols")
        return enhanced_features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def run_comprehensive_training(self, num_episodes: int = 1000, episode_length: int = 252) -> Dict[str, Any]:
        """Run comprehensive RL training with enhanced data."""
        
        logger.info(f"🚀 Starting comprehensive RL training: {num_episodes} episodes, {episode_length} steps each")
        
        self.training_stats['training_start'] = datetime.now()
        
        # 1. Prepare training data
        training_data = await self.prepare_comprehensive_training_data()
        self.training_data = training_data
        
        # 2. Initialize RL system
        logger.info("🤖 Initializing comprehensive RL system...")
        
        config = OnlineLearningConfig(
            buffer_size=50000,  # Larger buffer for more data
            batch_update_freq=pd.Timedelta(minutes=10),  # More frequent updates
            stable_policy_update_freq=pd.Timedelta(hours=2),
            safety_constraints=SafetyConstraints(
                max_drawdown=0.15,
                max_position_change=0.10,
                consecutive_losses_limit=5,
                volatility_threshold=0.05
            ),
            learner_performance_threshold=0.02,
            novelty_weight=0.3,
            uncertainty_weight=0.2
        )
        
        self.rl_system = create_comprehensive_rl_system(self.symbols, **config.__dict__)
        
        # 3. Create trading environment
        env = RealisticTradingEnvironment(
            symbols=self.symbols,
            initial_balance=100000,
            max_position_size=0.15
        )
        
        # 4. Training loop
        logger.info(f"🎯 Starting training loop...")
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # Reset environment for new episode
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            # Select random starting point in data
            available_symbols = list(training_data['combined_features'].keys())
            if not available_symbols:
                logger.error("❌ No training data available")
                break
                
            # Run episode
            for step in range(episode_length):
                try:
                    # Get current state from environment
                    observation = self._create_rl_observation(state, training_data, step)
                    
                    if observation is None:
                        break
                    
                    # Get RL action
                    regime_id = self._detect_regime(state, step)
                    action, metadata = self.rl_system.dual_agent.select_action(observation, regime_id)
                    
                    # Execute action in environment
                    next_state, reward, done, info = env.step(action)
                    
                    # Store experience
                    experience_data = {
                        'state': observation,
                        'action': action,
                        'reward': reward,
                        'next_state': self._create_rl_observation(next_state, training_data, step + 1),
                        'done': done,
                        'info': info
                    }
                    
                    # Add to RL system buffer
                    if hasattr(self.rl_system, 'store_experience'):
                        self.rl_system.store_experience(experience_data)
                    
                    # Update stats
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state
                    
                    if done:
                        break
                        
                except Exception as e:
                    logger.warning(f"Episode {episode} step {step} error: {e}")
                    break
            
            # Episode completed
            episode_duration = time.time() - episode_start
            
            # Update training stats
            self.training_stats['episodes'] += 1
            self.training_stats['total_steps'] += episode_steps
            self.training_stats['avg_rewards'].append(episode_reward)
            
            if episode_reward > self.training_stats['best_episode_reward']:
                self.training_stats['best_episode_reward'] = episode_reward
            
            # Periodic updates
            if episode % 10 == 0:
                # Trigger batch learning update
                if hasattr(self.rl_system, 'perform_batch_update'):
                    await self.rl_system.perform_batch_update()
                
                # Log progress
                recent_avg = np.mean(self.training_stats['avg_rewards'][-10:]) if len(self.training_stats['avg_rewards']) >= 10 else np.mean(self.training_stats['avg_rewards'])
                
                logger.info(f"Episode {episode:4d} | Steps: {episode_steps:3d} | Reward: {episode_reward:+8.4f} | Avg(10): {recent_avg:+8.4f} | Best: {self.training_stats['best_episode_reward']:+8.4f} | Duration: {episode_duration:.1f}s")
            
            # Save checkpoint every 50 episodes
            if episode % 50 == 0 and episode > 0:
                checkpoint_path = f"models/rl_checkpoint_episode_{episode}.pt"
                await self._save_training_checkpoint(checkpoint_path, episode)
        
        # Training completed
        training_duration = datetime.now() - self.training_stats['training_start']
        
        logger.info(f"🎉 Training completed!")
        logger.info(f"📊 Final stats:")
        logger.info(f"   Episodes: {self.training_stats['episodes']}")
        logger.info(f"   Total steps: {self.training_stats['total_steps']}")
        logger.info(f"   Best reward: {self.training_stats['best_episode_reward']:+.4f}")
        logger.info(f"   Avg reward: {np.mean(self.training_stats['avg_rewards']):+.4f}")
        logger.info(f"   Training time: {training_duration}")
        
        # Save final model
        final_model_path = f"models/comprehensive_rl_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        await self._save_final_model(final_model_path)
        
        return self.training_stats
    
    def _create_rl_observation(self, state: Dict, training_data: Dict, step: int) -> Optional[np.ndarray]:
        """Create RL observation from environment state and training data."""
        
        try:
            observation = []
            
            # Use consistent fixed number of symbols (already limited to 10 in _get_training_symbols)
            symbols_to_use = self.symbols  # All symbols (guaranteed to be exactly 10)
            
            for symbol in symbols_to_use:
                symbol_features = []
                
                # Get symbol data
                symbol_data = training_data['combined_features'].get(symbol)
                if symbol_data is not None and len(symbol_data) > step:
                    # Current market features (10 features per symbol)
                    current_row = symbol_data.iloc[min(step, len(symbol_data) - 1)]
                    
                    symbol_features = [
                        current_row.get('returns', 0.0),
                        current_row.get('volatility', 0.0),
                        current_row.get('rsi', 50.0) / 100.0,  # Normalize to 0-1
                        current_row.get('price_above_sma20', 0.0),
                        current_row.get('price_above_sma50', 0.0),
                        current_row.get('momentum_5', 0.0),
                        current_row.get('momentum_20', 0.0),
                        current_row.get('volume_ratio', 1.0),
                        current_row.get('vol_regime', 0.0),
                        current_row.get('trend_20', 0.0)
                    ]
                else:
                    # Default features if no data
                    symbol_features = [0.0] * 10
                
                observation.extend(symbol_features)
            
            # Ensure exact expected size (10 symbols * 10 features = 100)
            target_size = 10 * 10  # Must match RL system state_dim
            
            # Pad or truncate to exact target size
            if len(observation) < target_size:
                observation.extend([0.0] * (target_size - len(observation)))
            elif len(observation) > target_size:
                observation = observation[:target_size]
                
            # Verify final size
            assert len(observation) == target_size, f"Observation size mismatch: got {len(observation)}, expected {target_size}"
            
            return np.array(observation, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error creating RL observation: {e}")
            return None
    
    def _detect_regime(self, state: Dict, step: int) -> int:
        """Detect market regime for regime-conditional RL."""
        
        try:
            # Simple regime detection based on volatility and trends
            # 0: Low vol bull, 1: High vol bull, 2: Low vol bear, 3: High vol bear, 4: Sideways
            
            # This is simplified - in practice you'd use the enhanced features
            volatility = state.get('volatility', 0.02)
            trend = state.get('trend', 0.0)
            
            if volatility < 0.02:  # Low volatility
                if trend > 0.01:
                    return 0  # Low vol bull
                elif trend < -0.01:
                    return 2  # Low vol bear
                else:
                    return 4  # Sideways
            else:  # High volatility
                if trend > 0.01:
                    return 1  # High vol bull
                else:
                    return 3  # High vol bear
                    
        except Exception:
            return 0  # Default regime
    
    async def _save_training_checkpoint(self, filepath: str, episode: int):
        """Save training checkpoint."""
        
        try:
            checkpoint_data = {
                'episode': episode,
                'training_stats': self.training_stats,
                'model_state': 'saved',  # Simplified
                'timestamp': datetime.now().isoformat()
            }
            
            Path(filepath).parent.mkdir(exist_ok=True)
            
            with open(filepath.replace('.pt', '.json'), 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.info(f"💾 Checkpoint saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    async def _save_final_model(self, filepath: str):
        """Save final trained model."""
        
        try:
            final_stats = {
                'training_completed': datetime.now().isoformat(),
                'episodes_trained': self.training_stats['episodes'],
                'total_training_steps': self.training_stats['total_steps'],
                'best_episode_reward': self.training_stats['best_episode_reward'],
                'average_reward': np.mean(self.training_stats['avg_rewards']) if self.training_stats['avg_rewards'] else 0.0,
                'symbols_used': self.symbols,
                'data_sources': self.training_data.get('metadata', {}).get('data_sources', []),
                'model_architecture': 'comprehensive_rl_dual_agent'
            }
            
            Path(filepath).parent.mkdir(exist_ok=True)
            
            with open(filepath.replace('.pt', '.json'), 'w') as f:
                json.dump(final_stats, f, indent=2, default=str)
            
            # Save actual model state if available
            if hasattr(self.rl_system, 'save_system_state'):
                self.rl_system.save_system_state(filepath)
            
            logger.info(f"🎯 Final model saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")

async def main():
    """Main training execution."""
    
    logger.info("🚀 Comprehensive RL Training Starting...")
    
    trainer = ComprehensiveRLTrainer()
    
    # Run comprehensive training
    training_results = await trainer.run_comprehensive_training(
        num_episodes=500,     # Substantial training
        episode_length=63     # Quarter year of trading days
    )
    
    logger.info("✅ Comprehensive RL training completed!")
    
    return training_results

if __name__ == "__main__":
    # Create required directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    try:
        results = asyncio.run(main())
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Episodes: {results['episodes']}")
        print(f"📈 Best reward: {results['best_episode_reward']:+.4f}")
        print(f"⏱️  Duration: {datetime.now() - results['training_start']}")
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()