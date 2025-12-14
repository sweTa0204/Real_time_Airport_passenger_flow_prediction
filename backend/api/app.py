"""
Flask API for Airport Passenger Flow Prediction
Provides REST endpoints for predictions, data ingestion, and real-time alerts
"""

import os
import sys
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import threading
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.passenger_flow_model import PassengerFlowPredictor
from data.data_generator import AirportDataGenerator

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:5173"])
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
model = None
data_generator = None
current_data = None
alert_thresholds = {
    'security_wait': 25,
    'checkin_congestion': 15,
    'baggage_crowding': 500
}

def initialize_app():
    """Initialize the application with data and model"""
    global model, data_generator, current_data
    
    print("Initializing Airport Flow Prediction API...")
    
    # Initialize data generator
    data_generator = AirportDataGenerator(seed=42)
    
    # Check for saved model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'passenger_flow')
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = PassengerFlowPredictor.load(model_path)
    else:
        print("No saved model found. Generating data and training new model...")
        # Generate training data
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_data')
        data = data_generator.generate_all_data(
            output_dir=data_dir,
            historical_days=365,
            forecast_days=30
        )
        
        # Train model
        model = PassengerFlowPredictor(
            sequence_length=24,
            forecast_horizon=4,
            model_type='lstm'
        )
        model.train(data['historical_traffic'], epochs=30, verbose=1)
        model.save(model_path)
        
        current_data = data['historical_traffic']
    
    # Load current data if not set
    if current_data is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_data', 'historical_traffic.csv')
        if os.path.exists(data_path):
            current_data = pd.read_csv(data_path)
    
    print("API initialization complete!")


# ===== REST API Endpoints =====

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Get predictions for the next forecast horizon"""
    global model, current_data
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get input data
        if request.json and 'data' in request.json:
            df = pd.DataFrame(request.json['data'])
        elif current_data is not None:
            df = current_data.tail(100)
        else:
            return jsonify({'error': 'No data available'}), 400
        
        # Make prediction
        predictions = model.predict(df)
        
        # Generate alerts
        alerts = model.detect_crowd_surge(predictions, alert_thresholds)
        
        return jsonify({
            'predictions': predictions.to_dict(orient='records'),
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/current-status', methods=['GET'])
def get_current_status():
    """Get current airport status overview"""
    try:
        # Simulate current real-time data
        now = datetime.now()
        hour = now.hour
        
        # Base values vary by time of day
        base_traffic = {
            0: 50, 1: 30, 2: 20, 3: 20, 4: 40, 5: 100,
            6: 300, 7: 500, 8: 600, 9: 550, 10: 450, 11: 400,
            12: 400, 13: 450, 14: 500, 15: 550, 16: 600, 17: 650,
            18: 600, 19: 500, 20: 400, 21: 300, 22: 200, 23: 100
        }
        
        base = base_traffic.get(hour, 200)
        noise = np.random.uniform(0.85, 1.15)
        
        status = {
            'timestamp': now.isoformat(),
            'total_passengers': int(base * noise),
            'security': {
                'current_wait': round(max(5, np.random.normal(15, 5)), 1),
                'queue_length': int(max(10, np.random.normal(50, 20))),
                'throughput': int(np.random.uniform(120, 180)),
                'gates_open': np.random.randint(3, 6),
                'status': 'normal'
            },
            'checkin': {
                'current_wait': round(max(2, np.random.normal(8, 3)), 1),
                'counters_open': np.random.randint(10, 20),
                'kiosks_active': np.random.randint(15, 30),
                'status': 'normal'
            },
            'baggage': {
                'active_carousels': np.random.randint(3, 6),
                'passengers_waiting': int(np.random.uniform(20, 100)),
                'avg_wait': round(np.random.uniform(10, 25), 1),
                'status': 'normal'
            },
            'flights': {
                'departures_next_hour': np.random.randint(8, 20),
                'arrivals_next_hour': np.random.randint(8, 20),
                'delays': np.random.randint(0, 5),
                'cancellations': np.random.randint(0, 2)
            }
        }
        
        # Update status levels
        if status['security']['current_wait'] > 20:
            status['security']['status'] = 'busy'
        if status['security']['current_wait'] > 30:
            status['security']['status'] = 'critical'
            
        if status['checkin']['current_wait'] > 12:
            status['checkin']['status'] = 'busy'
        if status['checkin']['current_wait'] > 20:
            status['checkin']['status'] = 'critical'
            
        if status['baggage']['passengers_waiting'] > 80:
            status['baggage']['status'] = 'busy'
        if status['baggage']['passengers_waiting'] > 120:
            status['baggage']['status'] = 'critical'
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/historical', methods=['GET'])
def get_historical_data():
    """Get historical traffic data for charts"""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        # Generate historical data points
        now = datetime.now()
        data_points = []
        
        for i in range(hours, 0, -1):
            timestamp = now - timedelta(hours=i)
            hour = timestamp.hour
            
            base_traffic = {
                0: 50, 1: 30, 2: 20, 3: 20, 4: 40, 5: 100,
                6: 300, 7: 500, 8: 600, 9: 550, 10: 450, 11: 400,
                12: 400, 13: 450, 14: 500, 15: 550, 16: 600, 17: 650,
                18: 600, 19: 500, 20: 400, 21: 300, 22: 200, 23: 100
            }
            
            base = base_traffic.get(hour, 200)
            noise = np.random.uniform(0.85, 1.15)
            
            data_points.append({
                'timestamp': timestamp.isoformat(),
                'hour': hour,
                'passengers': int(base * noise),
                'security_wait': round(max(5, np.random.normal(15, 5)), 1),
                'checkin_wait': round(max(2, np.random.normal(8, 3)), 1),
                'baggage_wait': round(np.random.uniform(10, 25), 1)
            })
        
        return jsonify({'data': data_points})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/zones', methods=['GET'])
def get_zones():
    """Get all zone information"""
    zones = [
        {
            'id': 'security',
            'name': 'Security Checkpoints',
            'type': 'security',
            'locations': ['SG1', 'SG2', 'SG3', 'SG4', 'SG5']
        },
        {
            'id': 'checkin',
            'name': 'Check-in Counters',
            'type': 'checkin',
            'locations': [f'C{i}' for i in range(1, 21)]
        },
        {
            'id': 'baggage',
            'name': 'Baggage Claim',
            'type': 'baggage',
            'locations': ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6']
        },
        {
            'id': 'retail',
            'name': 'Retail & Dining',
            'type': 'retail',
            'locations': ['Terminal 1 Mall', 'Terminal 2 Food Court', 'Terminal 3 Shops']
        },
        {
            'id': 'lounge',
            'name': 'Lounges',
            'type': 'lounge',
            'locations': ['Business Lounge', 'First Class Lounge', 'Priority Pass Lounge']
        }
    ]
    
    return jsonify({'zones': zones})


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get current active alerts"""
    global model, current_data
    
    try:
        if model is None or current_data is None:
            # Return simulated alerts
            alerts = [
                {
                    'id': 'alert-1',
                    'type': 'crowd_surge',
                    'area': 'Security Wait',
                    'severity': 'medium',
                    'message': 'Expected security queue increase in 25 minutes',
                    'predicted_value': 22.5,
                    'threshold': 20,
                    'minutes_ahead': 25,
                    'timestamp': datetime.now().isoformat()
                }
            ]
        else:
            predictions = model.predict(current_data.tail(100))
            alerts = model.detect_crowd_surge(predictions, alert_thresholds)
            for i, alert in enumerate(alerts):
                alert['id'] = f'alert-{i+1}'
        
        return jsonify({'alerts': alerts})
    
    except Exception as e:
        return jsonify({'error': str(e), 'alerts': []}), 200


@app.route('/api/alerts/thresholds', methods=['GET', 'PUT'])
def manage_thresholds():
    """Get or update alert thresholds"""
    global alert_thresholds
    
    if request.method == 'GET':
        return jsonify({'thresholds': alert_thresholds})
    
    elif request.method == 'PUT':
        try:
            new_thresholds = request.json
            alert_thresholds.update(new_thresholds)
            return jsonify({'thresholds': alert_thresholds, 'message': 'Thresholds updated'})
        except Exception as e:
            return jsonify({'error': str(e)}), 400


@app.route('/api/flights', methods=['GET'])
def get_flights():
    """Get flight schedule information"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        # Generate sample flights
        airlines = ['Delta', 'United', 'American', 'Southwest', 'JetBlue']
        destinations = ['LAX', 'JFK', 'ORD', 'DFW', 'DEN', 'SFO', 'SEA', 'MIA']
        
        flights = []
        now = datetime.now()
        
        for i in range(limit):
            departure_time = now + timedelta(minutes=np.random.randint(15, 480))
            flights.append({
                'flight_id': f'{np.random.choice(airlines)[:2].upper()}{np.random.randint(100, 9999)}',
                'airline': np.random.choice(airlines),
                'destination': np.random.choice(destinations),
                'departure_time': departure_time.isoformat(),
                'gate': f'G{np.random.randint(1, 50)}',
                'terminal': f'T{np.random.randint(1, 4)}',
                'status': np.random.choice(['On Time', 'On Time', 'On Time', 'Delayed', 'Boarding']),
                'expected_passengers': np.random.randint(100, 300)
            })
        
        # Sort by departure time
        flights.sort(key=lambda x: x['departure_time'])
        
        return jsonify({'flights': flights})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Get passenger flow forecast for the next several hours"""
    global model, current_data
    
    try:
        hours = request.args.get('hours', 4, type=int)
        
        if model is not None and current_data is not None:
            predictions = model.predict(current_data.tail(100))
            forecast = predictions.to_dict(orient='records')
        else:
            # Generate simulated forecast
            now = datetime.now()
            forecast = []
            
            for i in range(hours):
                timestamp = now + timedelta(hours=i+1)
                hour = timestamp.hour
                
                base_traffic = {
                    0: 50, 1: 30, 2: 20, 3: 20, 4: 40, 5: 100,
                    6: 300, 7: 500, 8: 600, 9: 550, 10: 450, 11: 400,
                    12: 400, 13: 450, 14: 500, 15: 550, 16: 600, 17: 650,
                    18: 600, 19: 500, 20: 400, 21: 300, 22: 200, 23: 100
                }
                
                base = base_traffic.get(hour, 200)
                
                forecast.append({
                    'timestamp': timestamp.isoformat(),
                    'security_wait': round(max(5, np.random.normal(15 + base/100, 3)), 1),
                    'checkin_congestion': round(max(2, np.random.normal(8 + base/150, 2)), 1),
                    'baggage_crowding': int(base * np.random.uniform(0.3, 0.5))
                })
        
        return jsonify({'forecast': forecast})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== WebSocket Events for Real-time Updates =====

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Airport Flow Prediction API'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")


@socketio.on('subscribe_updates')
def handle_subscribe(data):
    """Subscribe to real-time updates"""
    print(f"Client {request.sid} subscribed to updates")
    emit('subscribed', {'status': 'subscribed'})


def broadcast_updates():
    """Background thread to broadcast updates"""
    while True:
        try:
            # Generate current status
            now = datetime.now()
            hour = now.hour
            
            base_traffic = {
                0: 50, 1: 30, 2: 20, 3: 20, 4: 40, 5: 100,
                6: 300, 7: 500, 8: 600, 9: 550, 10: 450, 11: 400,
                12: 400, 13: 450, 14: 500, 15: 550, 16: 600, 17: 650,
                18: 600, 19: 500, 20: 400, 21: 300, 22: 200, 23: 100
            }
            
            base = base_traffic.get(hour, 200)
            
            update = {
                'timestamp': now.isoformat(),
                'passengers': int(base * np.random.uniform(0.9, 1.1)),
                'security_wait': round(max(5, np.random.normal(15, 3)), 1),
                'checkin_wait': round(max(2, np.random.normal(8, 2)), 1),
                'baggage_wait': round(np.random.uniform(10, 20), 1)
            }
            
            socketio.emit('status_update', update)
            
            # Check for alerts
            if update['security_wait'] > alert_thresholds['security_wait']:
                socketio.emit('alert', {
                    'type': 'security_surge',
                    'message': f"High security wait time: {update['security_wait']} min",
                    'severity': 'high' if update['security_wait'] > 30 else 'medium',
                    'timestamp': now.isoformat()
                })
            
        except Exception as e:
            print(f"Error in broadcast: {e}")
        
        time.sleep(10)  # Update every 10 seconds


# Start background thread for broadcasts
def start_background_tasks():
    thread = threading.Thread(target=broadcast_updates, daemon=True)
    thread.start()


if __name__ == '__main__':
    # Initialize on startup
    initialize_app()
    
    # Start background tasks
    start_background_tasks()
    
    # Run server
    print("\n🚀 Starting Airport Flow Prediction API...")
    print("   REST API: http://localhost:5000/api")
    print("   WebSocket: ws://localhost:5000")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
