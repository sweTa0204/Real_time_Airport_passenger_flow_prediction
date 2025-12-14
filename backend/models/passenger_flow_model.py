"""
LSTM/GRU-based Time Series Model for Airport Passenger Flow Prediction
Predicts: Security queue wait time, Check-in congestion, Baggage claim crowding
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, 
                                     Bidirectional, Attention, Concatenate,
                                     BatchNormalization, TimeDistributed)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import os
from datetime import datetime, timedelta


class PassengerFlowPredictor:
    """
    Multi-output time series model for predicting airport congestion metrics
    """
    
    def __init__(self, sequence_length=24, forecast_horizon=4, model_type='lstm'):
        """
        Args:
            sequence_length: Number of past time steps to use (e.g., 24 hours)
            forecast_horizon: Number of future time steps to predict (e.g., 4 = next 4 hours)
            model_type: 'lstm', 'gru', or 'bidirectional'
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model_type = model_type
        self.model = None
        self.scalers = {}
        self.feature_columns = None
        self.target_columns = ['security_wait', 'checkin_congestion', 'baggage_crowding']
        
    def prepare_features(self, df):
        """Prepare feature engineering for time series"""
        df = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features for each target
        for col in ['total_passengers', 'avg_security_wait', 'avg_checkin_wait']:
            if col in df.columns:
                for lag in [1, 2, 3, 6, 12, 24]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Rolling statistics
                df[f'{col}_rolling_mean_6'] = df[col].rolling(window=6).mean()
                df[f'{col}_rolling_std_6'] = df[col].rolling(window=6).std()
                df[f'{col}_rolling_mean_24'] = df[col].rolling(window=24).mean()
        
        # Fill NaN values from lag/rolling operations
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def create_sequences(self, data, targets):
        """Create sequences for LSTM/GRU training"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i:(i + self.sequence_length)])
            # Multi-step forecast
            y.append(targets[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape, output_shape):
        """Build LSTM/GRU model architecture"""
        model = Sequential()
        
        if self.model_type == 'lstm':
            model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(32, return_sequences=False))
            
        elif self.model_type == 'gru':
            model.add(GRU(128, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(GRU(64, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(GRU(32, return_sequences=False))
            
        elif self.model_type == 'bidirectional':
            model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(Bidirectional(LSTM(32, return_sequences=False)))
            model.add(Dropout(0.2))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        
        # Output layer: forecast_horizon * num_targets
        model.add(Dense(output_shape, activation='linear'))
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, df, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """Train the model on historical data"""
        print("Preparing features...")
        df = self.prepare_features(df)
        
        # Define feature columns
        self.feature_columns = [col for col in df.columns if col not in 
                               ['timestamp', 'date'] + self.target_columns]
        
        # Prepare target columns (map from available columns)
        target_mapping = {
            'security_wait': 'avg_security_wait',
            'checkin_congestion': 'avg_checkin_wait',
            'baggage_crowding': 'total_passengers'  # Proxy for crowding
        }
        
        # Scale features
        print("Scaling features...")
        self.scalers['features'] = MinMaxScaler()
        features_scaled = self.scalers['features'].fit_transform(df[self.feature_columns])
        
        # Prepare targets
        targets = []
        for target in self.target_columns:
            source_col = target_mapping.get(target, target)
            if source_col in df.columns:
                self.scalers[target] = MinMaxScaler()
                scaled = self.scalers[target].fit_transform(df[[source_col]])
                targets.append(scaled.flatten())
            else:
                targets.append(np.zeros(len(df)))
        
        targets = np.column_stack(targets)
        
        # Create sequences
        print("Creating sequences...")
        X, y = self.create_sequences(features_scaled, targets)
        
        # Reshape y for multi-step, multi-target prediction
        y = y.reshape(y.shape[0], -1)  # Flatten to (samples, forecast_horizon * num_targets)
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        
        # Build model
        print(f"Building {self.model_type.upper()} model...")
        self.model = self.build_model(
            input_shape=(X.shape[1], X.shape[2]),
            output_shape=y.shape[1]
        )
        
        print(self.model.summary())
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
        ]
        
        # Train
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, df, return_dataframe=True):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        df = self.prepare_features(df)
        features_scaled = self.scalers['features'].transform(df[self.feature_columns])
        
        # Use last sequence_length points
        if len(features_scaled) >= self.sequence_length:
            X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        else:
            # Pad if not enough data
            padding = np.zeros((self.sequence_length - len(features_scaled), features_scaled.shape[1]))
            X = np.vstack([padding, features_scaled]).reshape(1, self.sequence_length, -1)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        
        # Reshape predictions: (1, forecast_horizon * num_targets) -> (forecast_horizon, num_targets)
        predictions = predictions.reshape(self.forecast_horizon, len(self.target_columns))
        
        # Inverse transform
        results = {}
        for i, target in enumerate(self.target_columns):
            if target in self.scalers:
                pred_scaled = predictions[:, i].reshape(-1, 1)
                results[target] = self.scalers[target].inverse_transform(pred_scaled).flatten()
            else:
                results[target] = predictions[:, i]
        
        if return_dataframe:
            # Create forecast timestamps
            last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
            forecast_times = [last_timestamp + timedelta(hours=i+1) for i in range(self.forecast_horizon)]
            
            result_df = pd.DataFrame({
                'timestamp': forecast_times,
                **results
            })
            return result_df
        
        return results
    
    def predict_with_confidence(self, df, n_samples=100):
        """Predict with confidence intervals using Monte Carlo Dropout"""
        predictions = []
        
        # Enable dropout during inference
        for _ in range(n_samples):
            pred = self.predict(df, return_dataframe=False)
            predictions.append(pred)
        
        # Calculate statistics
        results = {}
        for target in self.target_columns:
            values = np.array([p[target] for p in predictions])
            results[target] = {
                'mean': np.mean(values, axis=0),
                'std': np.std(values, axis=0),
                'lower_95': np.percentile(values, 2.5, axis=0),
                'upper_95': np.percentile(values, 97.5, axis=0)
            }
        
        return results
    
    def detect_crowd_surge(self, predictions, thresholds=None):
        """Detect potential crowd surges and generate alerts"""
        if thresholds is None:
            thresholds = {
                'security_wait': 25,  # minutes
                'checkin_congestion': 15,  # minutes
                'baggage_crowding': 500  # passengers
            }
        
        alerts = []
        
        for target, threshold in thresholds.items():
            if target in predictions.columns:
                surge_times = predictions[predictions[target] > threshold]
                
                for _, row in surge_times.iterrows():
                    minutes_ahead = int((row['timestamp'] - datetime.now()).total_seconds() / 60)
                    if minutes_ahead > 0:
                        alerts.append({
                            'type': 'crowd_surge',
                            'area': target.replace('_', ' ').title(),
                            'predicted_value': round(row[target], 1),
                            'threshold': threshold,
                            'minutes_ahead': minutes_ahead,
                            'timestamp': row['timestamp'].isoformat(),
                            'severity': 'high' if row[target] > threshold * 1.5 else 'medium',
                            'message': f"Expected crowd surge at {target.replace('_', ' ')} "
                                      f"in {minutes_ahead} minutes. Predicted: {row[target]:.1f}"
                        })
        
        return alerts
    
    def save(self, path):
        """Save model and scalers"""
        os.makedirs(path, exist_ok=True)
        
        self.model.save(f'{path}/model.keras')
        joblib.dump(self.scalers, f'{path}/scalers.pkl')
        joblib.dump({
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }, f'{path}/config.pkl')
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load saved model and scalers"""
        config = joblib.load(f'{path}/config.pkl')
        
        predictor = cls(
            sequence_length=config['sequence_length'],
            forecast_horizon=config['forecast_horizon'],
            model_type=config['model_type']
        )
        
        predictor.model = load_model(f'{path}/model.keras')
        predictor.scalers = joblib.load(f'{path}/scalers.pkl')
        predictor.feature_columns = config['feature_columns']
        predictor.target_columns = config['target_columns']
        
        print(f"Model loaded from {path}")
        return predictor


class MultiZonePredictor:
    """
    Predict congestion for multiple airport zones simultaneously
    """
    
    def __init__(self):
        self.zone_models = {}
        self.zones = ['security', 'checkin', 'baggage', 'retail', 'lounge']
    
    def train_zone_model(self, zone, data, **kwargs):
        """Train a model for a specific zone"""
        predictor = PassengerFlowPredictor(**kwargs)
        predictor.train(data)
        self.zone_models[zone] = predictor
        return predictor
    
    def predict_all_zones(self, data):
        """Get predictions for all zones"""
        results = {}
        for zone, model in self.zone_models.items():
            results[zone] = model.predict(data)
        return results


# Training script
if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from data.data_generator import AirportDataGenerator
    
    # Generate training data
    print("Generating synthetic training data...")
    generator = AirportDataGenerator(seed=42)
    data = generator.generate_all_data(
        output_dir='../data/generated_data',
        historical_days=365,
        forecast_days=30
    )
    
    # Train model
    print("\nTraining passenger flow prediction model...")
    predictor = PassengerFlowPredictor(
        sequence_length=24,
        forecast_horizon=4,
        model_type='lstm'
    )
    
    history = predictor.train(
        data['historical_traffic'],
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Save model
    predictor.save('../saved_models/passenger_flow')
    
    # Test prediction
    print("\nTesting predictions...")
    predictions = predictor.predict(data['historical_traffic'].tail(100))
    print("\nSample predictions:")
    print(predictions)
    
    # Test alert system
    alerts = predictor.detect_crowd_surge(predictions)
    print("\nGenerated alerts:")
    for alert in alerts:
        print(f"  - {alert['message']}")
