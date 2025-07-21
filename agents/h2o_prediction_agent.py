"""H2O.ai-powered time series prediction agent for stock price forecasting."""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import os
import tempfile

# Suppress H2O startup messages
warnings.filterwarnings('ignore')
os.environ['H2O_DISABLE_STRICT_VERSION_CHECK'] = '1'

try:
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator, H2ODeepLearningEstimator
    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("H2O.ai not available. Install with: pip install h2o")

from agents.market_analysis import (
    DataSource, DataReliability, DataPoint, AggregatedData, 
    multi_source_data, AnalysisResult, SignalStrength
)

logger = logging.getLogger(__name__)

@dataclass
class PricePrediction:
    """Stock price prediction with confidence intervals."""
    symbol: str
    current_price: float
    predicted_price_1d: float
    predicted_price_3d: float
    predicted_price_7d: float
    predicted_price_30d: float
    confidence_1d: float
    confidence_3d: float
    confidence_7d: float
    confidence_30d: float
    prediction_method: str
    features_used: List[str]
    model_accuracy: float
    timestamp: datetime
    price_direction: str  # "up", "down", "stable"
    volatility_forecast: float

class H2OPredictiveAnalysisAgent:
    """Advanced time series prediction agent using H2O.ai machine learning."""
    
    def __init__(self):
        """Initialize H2O.ai prediction agent."""
        self.name = "H2O.ai Time Series Prediction Agent"
        self.h2o_initialized = False
        self.models = {}
        self.feature_importance = {}
        self.prediction_cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
        if H2O_AVAILABLE:
            self._initialize_h2o()
        else:
            logger.error("H2O.ai not available - prediction agent disabled")
    
    def _initialize_h2o(self):
        """Initialize H2O cluster."""
        try:
            # Check if H2O is already running
            try:
                h2o.cluster().shutdown()
            except:
                pass
            
            # Start H2O with minimal configuration
            h2o.init(
                nthreads=-1,  # Use all available cores
                max_mem_size="4G",  # Limit memory usage
                strict_version_check=False,
                verbose=False
            )
            
            self.h2o_initialized = True
            logger.info("✅ H2O.ai cluster initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize H2O.ai: {e}")
            self.h2o_initialized = False
    
    def _prepare_time_series_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create comprehensive time series features for ML model."""
        try:
            # Ensure we have required columns
            if 'Close' not in df.columns:
                if 'close' in df.columns:
                    df = df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 
                                          'open': 'Open', 'volume': 'Volume'})
                else:
                    logger.error(f"Missing required price columns for {symbol}")
                    return pd.DataFrame()
            
            # Sort by date
            df = df.sort_index()
            
            # Price-based features
            df['returns_1d'] = df['Close'].pct_change()
            df['returns_3d'] = df['Close'].pct_change(periods=3)
            df['returns_7d'] = df['Close'].pct_change(periods=7)
            df['returns_30d'] = df['Close'].pct_change(periods=30)
            
            # Moving averages
            df['sma_5'] = df['Close'].rolling(5).mean()
            df['sma_10'] = df['Close'].rolling(10).mean()
            df['sma_20'] = df['Close'].rolling(20).mean()
            df['sma_50'] = df['Close'].rolling(50).mean()
            
            # Exponential moving averages
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()
            
            # Technical indicators
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['Close'].rolling(bb_period).mean()
            bb_std_dev = df['Close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility features
            df['volatility_10'] = df['returns_1d'].rolling(10).std()
            df['volatility_30'] = df['returns_1d'].rolling(30).std()
            
            # Price position features
            df['price_vs_sma20'] = df['Close'] / df['sma_20'] - 1
            df['price_vs_sma50'] = df['Close'] / df['sma_50'] - 1
            
            # Volume features (if available)
            if 'Volume' in df.columns and not df['Volume'].isna().all():
                df['volume_sma'] = df['Volume'].rolling(20).mean()
                df['volume_ratio'] = df['Volume'] / df['volume_sma']
                df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
            else:
                # Create dummy volume features if volume data not available
                df['volume_ratio'] = 1.0
                df['volume_trend'] = 1.0
            
            # High/Low features
            if 'High' in df.columns and 'Low' in df.columns:
                df['hl_ratio'] = df['High'] / df['Low'] - 1
                df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            else:
                df['hl_ratio'] = 0.0
                df['close_position'] = 0.5
            
            # Time-based features
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # Lagged features
            for lag in [1, 2, 3, 5, 7]:
                df[f'close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns_1d'].shift(lag)
                df[f'volume_ratio_lag_{lag}'] = df['volume_ratio'].shift(lag)
            
            # Target variables (future prices)
            df['target_1d'] = df['Close'].shift(-1)  # Next day's close
            df['target_3d'] = df['Close'].shift(-3)  # 3 days ahead
            df['target_7d'] = df['Close'].shift(-7)  # 1 week ahead
            df['target_30d'] = df['Close'].shift(-30)  # 1 month ahead
            
            # Remove rows with NaN values
            df = df.dropna()
            
            logger.info(f"Created {len(df.columns)} features for {symbol} with {len(df)} data points")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            return pd.DataFrame()
    
    def _train_prediction_models(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Train multiple H2O models for price prediction."""
        try:
            if not self.h2o_initialized or df.empty:
                return {}
            
            # Convert to H2O frame
            h2o_df = h2o.H2OFrame(df)
            
            # Define feature columns (exclude targets and non-predictive columns)
            exclude_columns = ['target_1d', 'target_3d', 'target_7d', 'target_30d', 'Close', 'High', 'Low', 'Open', 'Volume']
            feature_columns = [col for col in h2o_df.columns if col not in exclude_columns]
            
            # Split data for training (use 80% for training, 20% for validation)
            train_size = int(len(df) * 0.8)
            train_df = h2o_df[:train_size]
            valid_df = h2o_df[train_size:]
            
            models = {}
            
            # Train models for different prediction horizons
            for target in ['target_1d', 'target_3d', 'target_7d', 'target_30d']:
                if target not in h2o_df.columns:
                    continue
                
                logger.info(f"Training {target} prediction model for {symbol}")
                
                try:
                    # Gradient Boosting Model (primary)
                    gbm = H2OGradientBoostingEstimator(
                        ntrees=100,
                        max_depth=8,
                        learn_rate=0.1,
                        sample_rate=0.8,
                        col_sample_rate=0.8,
                        min_rows=5,
                        stopping_rounds=10,
                        stopping_tolerance=0.01,
                        stopping_metric="rmse",
                        seed=42,
                        model_id=f"gbm_{symbol}_{target}"
                    )
                    
                    gbm.train(
                        x=feature_columns,
                        y=target,
                        training_frame=train_df,
                        validation_frame=valid_df
                    )
                    
                    # Calculate model performance
                    train_rmse = gbm.rmse(train=True)
                    valid_rmse = gbm.rmse(valid=True)
                    
                    models[target] = {
                        'model': gbm,
                        'train_rmse': train_rmse,
                        'valid_rmse': valid_rmse,
                        'feature_importance': gbm.varimp(use_pandas=True),
                        'type': 'gbm'
                    }
                    
                    logger.info(f"✅ {target} model trained - RMSE: {valid_rmse:.4f}")
                    
                except Exception as model_error:
                    logger.warning(f"Failed to train {target} model: {model_error}")
                    
                    # Fallback to simpler Random Forest model
                    try:
                        rf = H2ORandomForestEstimator(
                            ntrees=50,
                            max_depth=10,
                            sample_rate=0.8,
                            col_sample_rate_per_tree=0.8,
                            seed=42,
                            model_id=f"rf_{symbol}_{target}"
                        )
                        
                        rf.train(
                            x=feature_columns,
                            y=target,
                            training_frame=train_df,
                            validation_frame=valid_df
                        )
                        
                        models[target] = {
                            'model': rf,
                            'train_rmse': rf.rmse(train=True),
                            'valid_rmse': rf.rmse(valid=True),
                            'feature_importance': rf.varimp(use_pandas=True),
                            'type': 'rf'
                        }
                        
                        logger.info(f"✅ Fallback {target} RF model trained")
                        
                    except Exception as rf_error:
                        logger.error(f"Both GBM and RF failed for {target}: {rf_error}")
            
            return models
            
        except Exception as e:
            logger.error(f"Error training models for {symbol}: {e}")
            return {}
    
    def _make_predictions(self, models: Dict[str, Any], latest_features: pd.DataFrame, 
                         symbol: str) -> PricePrediction:
        """Make price predictions using trained models."""
        try:
            if not models or latest_features.empty:
                current_price = latest_features.iloc[-1]['Close'] if 'Close' in latest_features.columns else 0
                return self._create_fallback_prediction(symbol, current_price)
            
            # Get the latest data point for prediction
            latest_row = latest_features.iloc[-1:].copy()
            
            # Remove target columns and non-predictive columns
            exclude_columns = ['target_1d', 'target_3d', 'target_7d', 'target_30d', 'Close', 'High', 'Low', 'Open', 'Volume']
            feature_columns = [col for col in latest_row.columns if col not in exclude_columns]
            
            # Convert to H2O frame
            h2o_latest = h2o.H2OFrame(latest_row[feature_columns])
            
            predictions = {}
            confidences = {}
            
            # Make predictions for each time horizon
            for target, model_info in models.items():
                try:
                    model = model_info['model']
                    pred = model.predict(h2o_latest)
                    prediction_value = pred.as_data_frame().iloc[0, 0]
                    
                    # Calculate confidence based on model performance
                    valid_rmse = model_info['valid_rmse']
                    current_price = latest_features.iloc[-1]['Close']
                    
                    # Confidence is inversely related to RMSE relative to current price
                    confidence = max(0.1, 1 - (valid_rmse / current_price))
                    confidence = min(confidence, 0.95)  # Cap at 95%
                    
                    predictions[target] = prediction_value
                    confidences[target] = confidence
                    
                except Exception as pred_error:
                    logger.warning(f"Prediction failed for {target}: {pred_error}")
                    # Use simple trend-based fallback
                    current_price = latest_features.iloc[-1]['Close']
                    recent_trend = latest_features['returns_1d'].iloc[-5:].mean()
                    
                    days_ahead = int(target.split('_')[1].replace('d', ''))
                    fallback_pred = current_price * (1 + recent_trend * days_ahead * 0.5)
                    
                    predictions[target] = fallback_pred
                    confidences[target] = 0.3
            
            # Extract predictions and confidences
            current_price = latest_features.iloc[-1]['Close']
            
            pred_1d = predictions.get('target_1d', current_price)
            pred_3d = predictions.get('target_3d', current_price)
            pred_7d = predictions.get('target_7d', current_price)
            pred_30d = predictions.get('target_30d', current_price)
            
            conf_1d = confidences.get('target_1d', 0.5)
            conf_3d = confidences.get('target_3d', 0.4)
            conf_7d = confidences.get('target_7d', 0.3)
            conf_30d = confidences.get('target_30d', 0.2)
            
            # Determine price direction
            price_changes = [
                (pred_1d - current_price) / current_price,
                (pred_3d - current_price) / current_price,
                (pred_7d - current_price) / current_price
            ]
            
            avg_change = np.mean(price_changes)
            if avg_change > 0.02:  # 2% threshold
                direction = "up"
            elif avg_change < -0.02:
                direction = "down"
            else:
                direction = "stable"
            
            # Calculate volatility forecast
            recent_volatility = latest_features['volatility_10'].iloc[-1]
            
            # Calculate model accuracy (average of all models)
            avg_accuracy = np.mean([1 - (info['valid_rmse'] / current_price) for info in models.values()])
            avg_accuracy = max(0.1, min(avg_accuracy, 0.95))
            
            # Get feature importance from best model
            best_model = min(models.values(), key=lambda x: x['valid_rmse'])
            features_used = best_model['feature_importance']['variable'][:10].tolist()
            
            return PricePrediction(
                symbol=symbol,
                current_price=current_price,
                predicted_price_1d=pred_1d,
                predicted_price_3d=pred_3d,
                predicted_price_7d=pred_7d,
                predicted_price_30d=pred_30d,
                confidence_1d=conf_1d,
                confidence_3d=conf_3d,
                confidence_7d=conf_7d,
                confidence_30d=conf_30d,
                prediction_method="h2o_ensemble",
                features_used=features_used,
                model_accuracy=avg_accuracy,
                timestamp=datetime.now(),
                price_direction=direction,
                volatility_forecast=recent_volatility
            )
            
        except Exception as e:
            logger.error(f"Error making predictions for {symbol}: {e}")
            current_price = latest_features.iloc[-1]['Close'] if not latest_features.empty else 0
            return self._create_fallback_prediction(symbol, current_price)
    
    def _create_fallback_prediction(self, symbol: str, current_price: float) -> PricePrediction:
        """Create a fallback prediction when ML models fail."""
        return PricePrediction(
            symbol=symbol,
            current_price=current_price,
            predicted_price_1d=current_price,
            predicted_price_3d=current_price,
            predicted_price_7d=current_price,
            predicted_price_30d=current_price,
            confidence_1d=0.3,
            confidence_3d=0.25,
            confidence_7d=0.2,
            confidence_30d=0.15,
            prediction_method="fallback",
            features_used=[],
            model_accuracy=0.3,
            timestamp=datetime.now(),
            price_direction="stable",
            volatility_forecast=0.2
        )
    
    async def predict_stock_price(self, symbol: str, periods: str = "2y") -> PricePrediction:
        """Main method to predict stock prices using H2O.ai models."""
        try:
            logger.info(f"Starting H2O.ai price prediction for {symbol}")
            
            if not self.h2o_initialized:
                logger.warning("H2O.ai not initialized - using fallback prediction")
                return self._create_fallback_prediction(symbol, 0)
            
            # Check cache first
            cache_key = f"{symbol}_{periods}"
            if cache_key in self.prediction_cache:
                cached_pred, cache_time = self.prediction_cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_duration:
                    logger.info(f"Using cached prediction for {symbol}")
                    return cached_pred
            
            # Get historical data for training
            stock_data = await multi_source_data.get_stock_data_multi_source(symbol, periods)
            
            if stock_data.confidence < 0.5 or stock_data.value.empty:
                logger.warning(f"Insufficient data for {symbol} prediction")
                return self._create_fallback_prediction(symbol, 0)
            
            df = stock_data.value.copy()
            
            # Prepare features
            df_features = self._prepare_time_series_features(df, symbol)
            
            if df_features.empty or len(df_features) < 100:
                logger.warning(f"Insufficient feature data for {symbol} (only {len(df_features)} points)")
                current_price = df.iloc[-1]['Close'] if not df.empty else 0
                return self._create_fallback_prediction(symbol, current_price)
            
            # Train models
            models = self._train_prediction_models(df_features, symbol)
            
            if not models:
                logger.warning(f"No models trained successfully for {symbol}")
                current_price = df_features.iloc[-1]['Close']
                return self._create_fallback_prediction(symbol, current_price)
            
            # Make predictions
            prediction = self._make_predictions(models, df_features, symbol)
            
            # Cache the prediction
            self.prediction_cache[cache_key] = (prediction, datetime.now())
            
            # Clean up H2O models to save memory
            for model_info in models.values():
                try:
                    h2o.remove(model_info['model'])
                except:
                    pass
            
            logger.info(f"✅ H2O.ai prediction completed for {symbol}: {prediction.price_direction} trend")
            return prediction
            
        except Exception as e:
            logger.error(f"H2O.ai prediction error for {symbol}: {e}")
            return self._create_fallback_prediction(symbol, 0)
    
    async def analyze_symbol(self, symbol: str) -> AnalysisResult:
        """Analyze symbol using H2O.ai predictions and convert to AnalysisResult."""
        try:
            logger.info(f"H2O.ai analysis for {symbol}")
            
            # Get price prediction
            prediction = await self.predict_stock_price(symbol)
            
            if prediction.model_accuracy < 0.3:
                return AnalysisResult(
                    symbol=symbol,
                    signal_type="h2o_prediction_low_confidence",
                    action="hold",
                    confidence=0.0,
                    strength=SignalStrength.VERY_WEAK,
                    reasoning="H2O.ai model accuracy too low for reliable predictions",
                    data_confidence=0.0
                )
            
            # Convert prediction to trading signal
            current_price = prediction.current_price
            
            # Use 7-day prediction for trading decisions (good balance of accuracy and relevance)
            predicted_price = prediction.predicted_price_7d
            confidence = prediction.confidence_7d
            
            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price
            
            # Generate trading signal
            if expected_return > 0.05 and confidence > 0.6:  # 5% upside with high confidence
                action = "buy"
                signal_confidence = confidence * min(expected_return / 0.1, 1.0)  # Scale by expected return
            elif expected_return < -0.05 and confidence > 0.6:  # 5% downside with high confidence
                action = "sell"
                signal_confidence = confidence * min(abs(expected_return) / 0.1, 1.0)
            elif expected_return > 0.02 and confidence > 0.4:  # 2% upside with medium confidence
                action = "buy"
                signal_confidence = confidence * 0.7
            elif expected_return < -0.02 and confidence > 0.4:  # 2% downside with medium confidence
                action = "sell"
                signal_confidence = confidence * 0.7
            else:
                action = "hold"
                signal_confidence = 0.5
            
            # Adjust confidence by model accuracy
            final_confidence = signal_confidence * prediction.model_accuracy
            
            # Determine signal strength
            if final_confidence >= 0.8:
                strength = SignalStrength.VERY_STRONG
            elif final_confidence >= 0.6:
                strength = SignalStrength.STRONG
            elif final_confidence >= 0.4:
                strength = SignalStrength.MODERATE
            elif final_confidence >= 0.2:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            # Calculate price targets based on predictions
            if action == "buy":
                price_target = prediction.predicted_price_7d
                stop_loss = current_price * (1 - prediction.volatility_forecast * 2)
            elif action == "sell":
                price_target = prediction.predicted_price_7d
                stop_loss = current_price * (1 + prediction.volatility_forecast * 2)
            else:
                price_target = None
                stop_loss = None
            
            # Build reasoning
            reasoning_parts = [
                f"H2O.ai ML prediction: {prediction.price_direction} trend",
                f"7-day target: ${predicted_price:.2f} ({expected_return:+.1%})",
                f"Model accuracy: {prediction.model_accuracy:.1%}",
                f"Volatility forecast: {prediction.volatility_forecast:.1%}"
            ]
            
            if prediction.features_used:
                reasoning_parts.append(f"Key features: {', '.join(prediction.features_used[:3])}")
            
            reasoning = ". ".join(reasoning_parts)
            
            # Create indicators dictionary
            indicators = {
                'current_price': current_price,
                'predicted_1d': prediction.predicted_price_1d,
                'predicted_3d': prediction.predicted_price_3d,
                'predicted_7d': prediction.predicted_price_7d,
                'predicted_30d': prediction.predicted_price_30d,
                'confidence_7d': prediction.confidence_7d,
                'expected_return_7d': expected_return,
                'model_accuracy': prediction.model_accuracy,
                'volatility_forecast': prediction.volatility_forecast
            }
            
            return AnalysisResult(
                symbol=symbol,
                signal_type="h2o_ml_prediction",
                action=action,
                confidence=final_confidence,
                strength=strength,
                price_target=price_target,
                stop_loss=stop_loss,
                reasoning=reasoning,
                indicators=indicators,
                data_sources=[DataSource.YAHOO_FINANCE],  # Primary data source
                data_confidence=prediction.model_accuracy,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"H2O.ai analysis error for {symbol}: {e}")
            return AnalysisResult(
                symbol=symbol,
                signal_type="h2o_error",
                action="hold",
                confidence=0.0,
                strength=SignalStrength.VERY_WEAK,
                reasoning=f"H2O.ai analysis error: {e}",
                data_confidence=0.0
            )
    
    def shutdown(self):
        """Shutdown H2O cluster and cleanup."""
        try:
            if self.h2o_initialized:
                h2o.cluster().shutdown()
                logger.info("H2O.ai cluster shutdown completed")
        except Exception as e:
            logger.warning(f"Error shutting down H2O.ai: {e}")

# Global H2O prediction agent
h2o_prediction_agent = H2OPredictiveAnalysisAgent() if H2O_AVAILABLE else None