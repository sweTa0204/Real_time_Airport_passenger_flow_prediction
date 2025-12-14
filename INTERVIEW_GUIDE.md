# 🎯 Interview Guide: Real-Time Airport Passenger Flow Prediction

## Quick Project Summary

**One-liner:** A full-stack ML application that predicts airport congestion using LSTM neural networks and displays real-time forecasts on a React dashboard.

**Tech Stack:** Python, TensorFlow/Keras, Flask, React, Socket.IO, Docker, Nginx

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Machine Learning Model](#3-machine-learning-model)
4. [Backend API](#4-backend-api)
5. [Frontend Dashboard](#5-frontend-dashboard)
6. [Docker & Deployment](#6-docker--deployment)
7. [Common Interview Questions](#7-common-interview-questions)
8. [Code Walkthrough](#8-code-walkthrough)
9. [What I Would Improve](#9-what-i-would-improve)

---

## 1. Project Overview

### Problem Statement
Airports face unpredictable passenger surges causing:
- Long security wait times
- Check-in counter congestion
- Baggage claim crowding

### Solution
A prediction system that:
1. **Forecasts** passenger flow 4 hours ahead using LSTM
2. **Alerts** staff before congestion occurs
3. **Displays** real-time metrics on a dashboard
4. **Recommends** optimal staff allocation (RL module)

### Key Metrics Predicted
| Metric | What It Measures | Alert Threshold |
|--------|------------------|-----------------|
| Security Wait | Minutes in queue | > 25 min |
| Check-in Congestion | Wait time at counters | > 15 min |
| Baggage Crowding | Passengers at carousel | > 500 |

---

## 2. Architecture Deep Dive

### System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     NGINX (Port 80)                             │
│  • Serves React static files                                    │
│  • Reverse proxy for /api/* → Flask                            │
│  • WebSocket proxy for Socket.IO                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│   REACT FRONTEND        │     │      FLASK BACKEND (Port 5000)  │
│   ─────────────────     │     │      ─────────────────────────  │
│   • Dashboard UI        │     │   REST API:                     │
│   • Recharts graphs     │     │   • /api/forecast               │
│   • Socket.IO client    │     │   • /api/current-status         │
│   • Real-time updates   │     │   • /api/alerts                 │
└─────────────────────────┘     │   • /api/predict                │
                                │                                 │
                                │   WebSocket:                    │
                                │   • Real-time status updates    │
                                │   • Push alerts                 │
                                │                                 │
                                │   ML Models:                    │
                                │   • LSTM Predictor              │
                                │   • RL Staff Scheduler (module) │
                                └─────────────────────────────────┘
```

### Data Flow
```
1. Historical Data → 2. Feature Engineering → 3. LSTM Model → 4. Predictions
                                                                    │
                                                                    ▼
7. User Dashboard ← 6. WebSocket Push ← 5. Alert Generation ← Threshold Check
```

---

## 3. Machine Learning Model

### Model Architecture: Stacked LSTM
```
Input: 24 hours of features (sequence_length=24)
              │
              ▼
┌─────────────────────────────┐
│  LSTM Layer 1 (128 units)   │  ← Learns long-term patterns
│  Dropout (0.2)              │
│  BatchNormalization         │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  LSTM Layer 2 (64 units)    │  ← Refines patterns
│  Dropout (0.2)              │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  LSTM Layer 3 (32 units)    │  ← Final encoding
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Dense (64) → Dense (32)    │
│  Dense (12) - Output        │  ← 4 hours × 3 metrics
└─────────────────────────────┘

Output: Next 4 hours predictions for 3 metrics
```

### Why LSTM?
- **Temporal Dependencies:** Remembers patterns over time (8 AM today similar to 8 AM yesterday)
- **Sequential Data:** Airport traffic is inherently time-series
- **Long-term Memory:** Can capture weekly patterns (Monday vs Sunday)

### Feature Engineering

#### 1. Cyclical Encoding (Time Features)
**Problem:** Hour 23 and Hour 0 are adjacent but appear far apart (23 vs 0)

**Solution:** Encode as sin/cos to preserve circular nature
```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

#### 2. Lag Features
Past values as predictors:
| Lag | Hours Ago | What It Captures |
|-----|-----------|------------------|
| lag_1 | 1 hour | Immediate trend |
| lag_6 | 6 hours | Half-day cycle |
| lag_24 | 24 hours | **Same time yesterday** |

#### 3. Rolling Statistics
```python
df['rolling_mean_6'] = df['passengers'].rolling(6).mean()   # Smoothed trend
df['rolling_std_6'] = df['passengers'].rolling(6).std()     # Volatility
```

### Model Training
```python
# Hyperparameters
sequence_length = 24      # Use last 24 hours
forecast_horizon = 4      # Predict next 4 hours
epochs = 50
batch_size = 32
optimizer = Adam(lr=0.001)
loss = 'mse'

# Callbacks
- EarlyStopping(patience=15)      # Stop if no improvement
- ReduceLROnPlateau(factor=0.5)   # Lower learning rate when stuck
- ModelCheckpoint()                # Save best model
```

### Prediction with Confidence
```python
def predict_with_confidence(self, df, n_samples=100):
    # Monte Carlo Dropout - run 100 predictions with dropout ON
    # Returns: mean, std, 95% confidence interval
```

---

## 4. Backend API

### Technology: Flask + Flask-SocketIO

### REST Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check, model status |
| `/api/current-status` | GET | Current airport metrics |
| `/api/forecast` | GET | ML predictions for next N hours |
| `/api/predict` | POST | Custom prediction with input data |
| `/api/alerts` | GET | Active congestion alerts |
| `/api/historical` | GET | Past traffic data for charts |
| `/api/flights` | GET | Flight schedule information |
| `/api/zones` | GET | Airport zone details |

### WebSocket Events (Socket.IO)
```
Server → Client:
- 'status_update': Real-time metrics every 10 seconds
- 'alert': Push notification when threshold exceeded

Client → Server:
- 'subscribe_updates': Request real-time stream
```

### Global State
```python
model = None              # Loaded LSTM model (shared across requests)
data_generator = None     # Synthetic data generator
current_data = None       # Recent traffic data for predictions
alert_thresholds = {      # Configurable via API
    'security_wait': 25,
    'checkin_congestion': 15,
    'baggage_crowding': 500
}
```

### Why Global State?
- **Performance:** Load model once (~2 sec) vs per-request (~2 sec each)
- **Memory:** One model instance serves all requests
- **Limitation:** Doesn't scale with multiple workers (use Redis in production)

---

## 5. Frontend Dashboard

### Technology: React 18 + Vite + TailwindCSS

### Key Components
```
src/
├── App.jsx              # Main component, data fetching
├── components/
│   ├── Dashboard.jsx    # Main dashboard layout
│   ├── Header.jsx       # Navigation, status indicator
│   └── AlertPanel.jsx   # Real-time alerts display
└── services/
    └── api.js           # API client functions
```

### Data Fetching Strategy
```javascript
// Initial load - REST API
useEffect(() => {
  fetchCurrentStatus();   // GET /api/current-status
  fetchForecast();        // GET /api/forecast
  fetchAlerts();          // GET /api/alerts
}, []);

// Real-time updates - WebSocket
useEffect(() => {
  const socket = io('/');
  socket.on('status_update', (data) => setStatus(data));
  socket.on('alert', (alert) => addAlert(alert));
}, []);
```

### Visualization: Recharts
- Line charts for historical trends
- Area charts for forecasts
- Bar charts for zone comparisons

---

## 6. Docker & Deployment

### Container Architecture
```
docker-compose.yml
├── frontend (nginx:alpine)
│   └── Serves React build + proxies API
└── backend (python:3.11-slim)
    └── Flask + TensorFlow + ML model
```

### Dockerfile.frontend (Multi-stage Build)
```dockerfile
# Stage 1: Build React app
FROM node:18-alpine AS builder
RUN npm install && npm run build

# Stage 2: Serve with Nginx (tiny image)
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
```
**Result:** 50MB image instead of 1GB+

### Dockerfile.backend
```dockerfile
FROM python:3.11-slim
RUN pip install -r requirements.txt  # TensorFlow, Flask, etc.
CMD ["python", "api/app.py"]
```

### Nginx Configuration
```nginx
location / {
    # Serve React static files
    try_files $uri $uri/ /index.html;  # SPA routing support
}

location /api/ {
    # Proxy to Flask backend
    proxy_pass http://backend:5000/api/;
}

location /socket.io/ {
    # WebSocket upgrade for real-time
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

### Running the Project
```bash
# Build and start
docker-compose up --build

# Access
Frontend: http://localhost
Backend API: http://localhost:5001/api/health
```

---

## 7. Common Interview Questions

### ML Questions

**Q: Why did you choose LSTM over other models?**
> LSTM excels at sequential data with temporal dependencies. Airport traffic has strong daily/weekly patterns - LSTM's memory cells capture "8 AM Monday" being similar to "8 AM last Monday". I also implemented GRU as an alternative (faster training, fewer parameters).

**Q: How do you handle seasonality?**
> Two approaches: (1) Cyclical encoding - convert hour/day to sin/cos pairs so 23:00 and 00:00 are close, (2) Lag features - include "same time yesterday" (lag_24) as a feature.

**Q: What's your model's accuracy?**
> I use MSE (Mean Squared Error) and MAE (Mean Absolute Error) as metrics. The exact values depend on the data, but more importantly, I implemented confidence intervals using Monte Carlo Dropout to quantify prediction uncertainty.

**Q: How would you improve the model?**
> (1) Add external data - flight schedules, weather, holidays
> (2) Attention mechanism - focus on relevant time steps
> (3) Ensemble - combine LSTM with gradient boosting
> (4) Online learning - update weights with streaming data

### Backend Questions

**Q: Why Flask instead of FastAPI?**
> Flask with Flask-SocketIO provides mature WebSocket support. FastAPI is faster for pure REST, but Socket.IO integration is simpler with Flask. For a larger system, I'd consider FastAPI + WebSocket or even a dedicated real-time service.

**Q: How do you handle concurrent requests?**
> Flask-SocketIO runs with threading mode. The ML model is loaded once and shared (global state). For production scale, I'd use Gunicorn workers + Redis for session storage.

**Q: Explain the WebSocket vs REST decision.**
> REST for: Initial data load, on-demand queries, configuration updates
> WebSocket for: Real-time dashboard updates (every 10 sec), instant alerts
> This hybrid approach minimizes unnecessary polling while enabling real-time features.

### DevOps Questions

**Q: Why Docker?**
> (1) Reproducibility - anyone can run with single command
> (2) Isolation - TensorFlow version locked, no conflicts
> (3) Deployment - same image runs locally and on AWS/Azure
> (4) Microservices - frontend/backend can scale independently

**Q: Explain your Nginx configuration.**
> Nginx serves three roles: (1) Static file server for React build, (2) Reverse proxy routing /api to Flask, (3) WebSocket proxy with upgrade headers. The `try_files` directive enables React Router's client-side routing.

**Q: How would you deploy to production?**
> I created Azure Container Apps configuration:
> - Container Registry for images
> - Container Apps for serverless containers
> - GitHub Actions for CI/CD
> - Could also use AWS ECS or Kubernetes

### System Design Questions

**Q: How would you scale this for a real airport?**
```
Current (Demo):                    Production Scale:
Static CSV → Model                 Kafka → Stream Processing → Model
                                   
                                   ┌─────────────────────────┐
IoT Sensors ──► Kafka ──► Flink ──►│ Model Serving (TF Serve)│
                                   └─────────────────────────┘
                                              │
                                   ┌──────────┴──────────┐
                                   ▼                     ▼
                              TimescaleDB            Redis Cache
                                   │                     │
                                   └──────────┬──────────┘
                                              ▼
                                        Flask API (multiple replicas)
                                              │
                                        Load Balancer
                                              │
                                         Clients
```

**Q: What's missing for production?**
> (1) Real data pipeline - Kafka for sensor ingestion
> (2) Model versioning - MLflow for experiment tracking
> (3) Monitoring - Prometheus + Grafana for metrics
> (4) Retraining pipeline - Airflow for scheduled retraining
> (5) A/B testing - compare model versions in production

---

## 8. Code Walkthrough

### Key Files to Know

#### `backend/models/passenger_flow_model.py`
```python
class PassengerFlowPredictor:
    def __init__(self, sequence_length=24, forecast_horizon=4, model_type='lstm'):
        # Configuration
    
    def prepare_features(self, df):
        # Cyclical encoding, lag features, rolling stats
    
    def build_model(self, input_shape, output_shape):
        # LSTM/GRU architecture
    
    def train(self, df, epochs=100):
        # Model training with callbacks
    
    def predict(self, df):
        # Inference on new data
    
    def save(self, path) / load(cls, path):
        # Persist model + scalers + config
```

#### `backend/api/app.py`
```python
# Global state
model = None  # Shared across requests

def initialize_app():
    # Load or train model at startup

@app.route('/api/forecast')
def get_forecast():
    predictions = model.predict(current_data.tail(100))
    return jsonify({'forecast': predictions})

@socketio.on('connect')
def handle_connect():
    # WebSocket connection handling

def broadcast_updates():
    # Background thread pushing updates every 10 sec
```

#### `frontend/src/App.jsx`
```javascript
function App() {
  const [status, setStatus] = useState(null);
  const [alerts, setAlerts] = useState([]);
  
  useEffect(() => {
    // REST API calls for initial data
    fetchCurrentStatus();
    
    // WebSocket for real-time updates
    const socket = io('/');
    socket.on('status_update', setStatus);
    socket.on('alert', (a) => setAlerts(prev => [...prev, a]));
  }, []);
}
```

---

## 9. What I Would Improve

### Short-term Improvements
1. **Integrate RL Staff Scheduler** - Connect the PPO module to API
2. **Add authentication** - JWT tokens for API security
3. **Implement caching** - Redis for frequent queries
4. **Add unit tests** - pytest for backend, Jest for frontend

### Long-term / Production Improvements
1. **Real data pipeline** - Kafka + Flink for streaming
2. **Model serving** - TensorFlow Serving for scalability
3. **Monitoring** - Prometheus metrics, Grafana dashboards
4. **MLOps** - MLflow for experiment tracking, model registry
5. **Kubernetes** - For orchestration and auto-scaling

### If Asked "What Would You Do Differently?"
> "For a production system, I would:
> 1. Use TensorFlow Serving instead of loading model in Flask
> 2. Add a message queue (Kafka) for real-time data ingestion
> 3. Implement proper CI/CD with model validation tests
> 4. Add comprehensive monitoring and alerting
> 5. Consider a Transformer architecture for potentially better accuracy"

---

## 🎯 Quick Reference Card

### Project in 30 Seconds
> "I built a real-time airport congestion prediction system. It uses an LSTM neural network to forecast security wait times, check-in queues, and baggage claim crowding 4 hours ahead. The backend is Flask with REST and WebSocket APIs, frontend is React with live updating charts, all containerized with Docker."

### Tech Stack One-liner
> "Python, TensorFlow/Keras, Flask, Socket.IO, React, Docker, Nginx"

### Three Impressive Technical Details
1. **Cyclical time encoding** - sin/cos transformation for temporal features
2. **Monte Carlo Dropout** - confidence intervals for predictions
3. **Multi-stage Docker build** - 50MB frontend image vs 1GB

### Two Honest Limitations
1. Uses simulated data (production would need real sensor feeds)
2. RL staff scheduler is modular but not integrated into API

---

## Good Luck! 🚀

Remember:
- Be honest about what's simulated vs production-ready
- Show enthusiasm for the ML concepts
- Mention improvements you'd make
- Relate to their airport domain if possible

