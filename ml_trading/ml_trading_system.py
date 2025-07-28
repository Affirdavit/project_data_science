# ML Algorithmic Trading System
# A comprehensive template for machine learning-based algorithmic trading
# Updated for 2025 with modern Python tools and best practices

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score
import ta
import warnings
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import sqlite3
import asyncio
import ccxt  # For crypto exchanges
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Configuration class for trading parameters"""
    symbols: List[str]
    lookback_period: int = 252  # Days
    feature_window: int = 20    # Days for technical indicators
    prediction_horizon: int = 5 # Days ahead to predict
    train_test_split: float = 0.8
    risk_free_rate: float = 0.02
    transaction_cost: float = 0.001  # 0.1%
    max_position_size: float = 0.1   # 10% of portfolio
    rebalance_frequency: str = 'weekly'  # daily, weekly, monthly

class DataManager:
    """Handles data ingestion, storage, and retrieval"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing market data"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                adj_close REAL,
                PRIMARY KEY (symbol, date)
            )
        ''')
        conn.commit()
        conn.close()
    
    def fetch_data(self, symbols: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """Fetch market data for given symbols"""
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                
                # Standardize column names to lowercase
                df.columns = [col.lower() for col in df.columns]
                
                # Reset index to get Date as a column
                df.reset_index(inplace=True)
                df.columns = [col.lower() for col in df.columns]
                
                # Ensure we have the required columns
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Missing required columns for {symbol}. Available: {df.columns.tolist()}")
                    continue
                
                df['symbol'] = symbol
                data[symbol] = df
                logger.info(f"Fetched data for {symbol}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        return data
    
    def store_data(self, data: Dict[str, pd.DataFrame]):
        """Store market data in database"""
        conn = sqlite3.connect(self.db_path)
        for symbol, df in data.items():
            # Ensure we have the right column names
            df_store = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            df_store['adj_close'] = df['close']  # Use close as adj_close if not available
            df_store['symbol'] = symbol
            df_store.to_sql('market_data', conn, if_exists='replace', index=False)
        conn.close()
    
    def load_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load market data from database"""
        conn = sqlite3.connect(self.db_path)
        data = {}
        for symbol in symbols:
            query = f"SELECT * FROM market_data WHERE symbol = '{symbol}' ORDER BY date"
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                data[symbol] = df
        conn.close()
        return data

class FeatureEngineer:
    """Creates technical indicators and features for ML models"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical analysis features"""
        data = df.copy()
        
        # Handle missing data
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f'sma_{window}'] = ta.trend.sma_indicator(data['close'], window=window)
            data[f'ema_{window}'] = ta.trend.ema_indicator(data['close'], window=window)
        
        # Price ratios
        data['price_to_sma20'] = data['close'] / data['sma_20']
        data['price_to_sma50'] = data['close'] / data['sma_50']
        
        # Volatility indicators
        data['bb_upper'] = ta.volatility.bollinger_hband(data['close'], window=20)
        data['bb_lower'] = ta.volatility.bollinger_lband(data['close'], window=20)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['close']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Momentum indicators
        data['rsi'] = ta.momentum.rsi(data['close'], window=14)
        
        # MACD with error handling
        try:
            data['macd'] = ta.trend.macd_diff(data['close'])
            data['macd_signal'] = ta.trend.macd_signal(data['close'])
        except:
            data['macd'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Stochastic oscillator
        try:
            data['stoch'] = ta.momentum.stoch(data['high'], data['low'], data['close'], window=14)
        except:
            data['stoch'] = ((data['close'] - data['low'].rolling(14).min()) / 
                           (data['high'].rolling(14).max() - data['low'].rolling(14).min()) * 100)
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # On Balance Volume
        try:
            data['obv'] = ta.volume.on_balance_volume(data['close'], data['volume'])
        except:
            data['obv'] = (data['volume'] * np.sign(data['close'].diff())).cumsum()
        
        # Trend indicators
        try:
            data['adx'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14)
        except:
            data['adx'] = 50  # Default neutral value
            
        try:
            data['cci'] = ta.trend.cci(data['high'], data['low'], data['close'], window=20)
        except:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            data['cci'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).std())
        
        # Statistical features
        data['volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        data['skewness'] = data['returns'].rolling(window=20).skew()
        data['kurtosis'] = data['returns'].rolling(window=20).kurt()
        
        # Market structure features
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_to_high'] = data['close'] / data['high']
        data['close_to_low'] = data['close'] / data['low']
        
        # Fill any remaining NaN values
        data = data.fillna(method='ffill').fillna(0)
        
        return data
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target labels for classification"""
        data = df.copy()
        
        # Forward returns
        data['forward_returns'] = data['close'].shift(-self.config.prediction_horizon) / data['close'] - 1
        
        # Classification labels based on forward returns
        data['target'] = 0  # Hold
        data.loc[data['forward_returns'] > 0.02, 'target'] = 1  # Buy (>2% return)
        data.loc[data['forward_returns'] < -0.02, 'target'] = -1  # Sell (<-2% return)
        
        # Alternative: Regime-based labels
        data['volatility_regime'] = pd.qcut(data['volatility'].fillna(data['volatility'].median()), 
                                          q=3, labels=['low', 'medium', 'high'])
        
        return data

class MLModel:
    """Machine learning model for trading predictions"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for ML model"""
        # Exclude non-numeric and label columns
        exclude_cols = ['target', 'forward_returns', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        df_clean = df[feature_cols + ['target']].dropna()

        # Safety: drop any rows with non-numeric values
        df_clean = df_clean.apply(pd.to_numeric, errors='coerce').dropna()

        X = df_clean[feature_cols].values
        y = df_clean['target'].values

        self.feature_columns = feature_cols

        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray, symbol: str):
        """Train multiple ML models"""
        # Time series split for training
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[symbol] = scaler
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        trained_models = {}
        for name, model in models.items():
            logger.info(f"Training {name} for {symbol}")
            
            if name == 'logistic_regression':
                model.fit(X_scaled, y)
                trained_models[name] = model
            else:
                model.fit(X, y)
                trained_models[name] = model
            
            # Cross-validation score
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if name == 'logistic_regression':
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)
                
                model_cv = models[name]
                model_cv.fit(X_train, y_train)
                score = model_cv.score(X_val, y_val)
                scores.append(score)
            
            logger.info(f"{name} CV Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        self.models[symbol] = trained_models
    
    def predict(self, X: np.ndarray, symbol: str, model_name: str = 'random_forest') -> np.ndarray:
        """Make predictions using trained model"""
        if symbol not in self.models:
            raise ValueError(f"No trained model found for {symbol}")
        
        model = self.models[symbol][model_name]
        
        if model_name == 'logistic_regression':
            scaler = self.scalers[symbol]
            X_scaled = scaler.transform(X)
            return model.predict(X_scaled)
        else:
            return model.predict(X)
    
    def get_feature_importance(self, symbol: str, model_name: str = 'random_forest') -> pd.DataFrame:
        """Get feature importance from trained model"""
        if symbol not in self.models:
            return pd.DataFrame()
        
        model = self.models[symbol][model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return pd.DataFrame()

class BacktestEngine:
    """Backtesting engine for strategy evaluation"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.trades = []
        self.portfolio_values = []
    
    def run_backtest(self, predictions: Dict[str, pd.DataFrame], 
                    market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run backtest simulation"""
        results = {}
        initial_capital = 100000
        current_capital = initial_capital
        positions = {}
        
        # Convert market_data to proper format if needed
        market_data_indexed = {}
        for symbol, df in market_data.items():
            if 'date' in df.columns:
                df_indexed = df.set_index('date')
            else:
                df_indexed = df
            market_data_indexed[symbol] = df_indexed
        
        # Align dates across all symbols
        all_dates = set()
        for symbol_data in market_data_indexed.values():
            all_dates.update(symbol_data.index)
        all_dates = sorted(list(all_dates))
        
        portfolio_history = []
        
        for date in all_dates:
            portfolio_value = current_capital
            
            # Check for signals and execute trades
            for symbol in self.config.symbols:
                if (symbol in predictions and 
                    date in predictions[symbol].index and 
                    symbol in market_data_indexed and 
                    date in market_data_indexed[symbol].index):
                    
                    signal = predictions[symbol].loc[date, 'predictions']
                    price = market_data_indexed[symbol].loc[date, 'close']
                    
                    # Position sizing based on signal strength
                    if signal == 1:  # Buy signal
                        position_size = min(self.config.max_position_size * current_capital, 
                                          current_capital * 0.95) / price
                        if symbol not in positions:
                            positions[symbol] = 0
                        positions[symbol] += position_size
                        current_capital -= position_size * price * (1 + self.config.transaction_cost)
                    
                    elif signal == -1 and symbol in positions and positions[symbol] > 0:  # Sell signal
                        position_value = positions[symbol] * price * (1 - self.config.transaction_cost)
                        current_capital += position_value
                        positions[symbol] = 0
            
            # Calculate portfolio value
            for symbol, position in positions.items():
                if (symbol in market_data_indexed and 
                    date in market_data_indexed[symbol].index and 
                    position > 0):
                    portfolio_value += position * market_data_indexed[symbol].loc[date, 'close']
            
            portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': current_capital
            })
        
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        results = {
            'total_return': (portfolio_df['portfolio_value'].iloc[-1] / initial_capital - 1) * 100,
            'annualized_return': ((portfolio_df['portfolio_value'].iloc[-1] / initial_capital) ** 
                                 (252 / len(portfolio_df)) - 1) * 100,
            'volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': (returns.mean() * 252 - self.config.risk_free_rate) / 
                           (returns.std() * np.sqrt(252)),
            'max_drawdown': self.calculate_max_drawdown(portfolio_df['portfolio_value']),
            'portfolio_history': portfolio_df
        }
        
        return results
    
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min() * 100

class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer(config)
        self.ml_model = MLModel(config)
        self.backtest_engine = BacktestEngine(config)
    
    def run_full_pipeline(self):
        """Execute the complete ML trading pipeline"""
        logger.info("Starting ML Trading System Pipeline")
        
        # 1. Data ingestion
        logger.info("Step 1: Data Ingestion")
        market_data = self.data_manager.fetch_data(self.config.symbols)
        self.data_manager.store_data(market_data)
        
        # 2. Feature engineering
        logger.info("Step 2: Feature Engineering")
        processed_data = {}
        for symbol, df in market_data.items():
            # Set date as index for technical analysis
            df_indexed = df.set_index('date')
            
            # Create technical features
            df_features = self.feature_engineer.create_technical_features(df_indexed)
            # Create labels
            df_labels = self.feature_engineer.create_labels(df_features)
            processed_data[symbol] = df_labels
        
        # 3. Model training
        logger.info("Step 3: Model Training")
        predictions = {}
        for symbol, df in processed_data.items():
            logger.info(f"Training models for {symbol}")
            
            # Prepare features
            X, y = self.ml_model.prepare_features(df)
            
            # Train/test split (time-aware)
            split_idx = int(len(X) * self.config.train_test_split)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train models
            self.ml_model.train_models(X_train, y_train, symbol)
            
            # Make predictions
            y_pred = self.ml_model.predict(X_test, symbol)
            
            # Store predictions with dates
            test_dates = df.index[split_idx:len(X)+split_idx]
            pred_df = pd.DataFrame({
                'predictions': y_pred,
                'actual': y_test
            }, index=test_dates)
            predictions[symbol] = pred_df
            
            # Model evaluation
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy for {symbol}: {accuracy:.4f}")
            
            # Feature importance
            importance_df = self.ml_model.get_feature_importance(symbol)
            logger.info(f"Top 5 features for {symbol}:")
            logger.info(importance_df.head().to_string())
        
        # 4. Backtesting
        logger.info("Step 4: Backtesting")
        backtest_results = self.backtest_engine.run_backtest(predictions, market_data)
        
        # 5. Results reporting
        logger.info("Step 5: Results Summary")
        self.print_results(backtest_results)
        
        return {
            'processed_data': processed_data,
            'predictions': predictions,
            'backtest_results': backtest_results
        }
    
    def print_results(self, results: Dict):
        """Print comprehensive results"""
        print("\n" + "="*50)
        print("ML TRADING SYSTEM RESULTS")
        print("="*50)
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Annualized Return: {results['annualized_return']:.2f}%")
        print(f"Volatility: {results['volatility']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
        print("="*50)

# Example usage and configuration
if __name__ == "__main__":
    # Configure the trading system
    config = TradingConfig(
        symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        lookback_period=504,  # 2 years
        feature_window=20,
        prediction_horizon=5,
        train_test_split=0.8,
        max_position_size=0.15,
        transaction_cost=0.001
    )
    
    # Initialize and run the trading system
    trading_system = TradingSystem(config)
    
    try:
        results = trading_system.run_full_pipeline()
        
        # Optional: Save models for production use
        for symbol in config.symbols:
            if symbol in trading_system.ml_model.models:
                model_path = f"models/{symbol}_trading_model.joblib"
                joblib.dump(trading_system.ml_model.models[symbol], model_path)
                logger.info(f"Saved model for {symbol} to {model_path}")
        
        # Optional: Plot portfolio performance
        portfolio_history = results['backtest_results']['portfolio_history']
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_history.index, portfolio_history['portfolio_value'])
        plt.title('Portfolio Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in trading system execution: {e}")
        raise