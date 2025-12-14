# ✈️ Real-Time Airport Passenger Flow Prediction

An end-to-end AI system for predicting and managing airport congestion at security gates, check-in counters, lounges, and retail zones.

![Dashboard Preview](https://img.shields.io/badge/Status-Active-green) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![React](https://img.shields.io/badge/React-18-61dafb) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

## 🎯 Features

- **LSTM/GRU Time Series Forecasting**: Predict security wait times, check-in congestion, and baggage claim crowding
- **Real-Time Dashboard**: React-based visualization with live updates via WebSocket
- **Smart Alerts**: "Expected crowd surge in 20 minutes" notifications
- **Synthetic Data Generation**: Realistic airport traffic patterns for training
- **RL Staff Scheduling** (Optional): Reinforcement learning for optimal staff allocation
- **Docker Deployment**: Ready for Azure/AWS deployment

## 📁 Project Structure

```
├── backend/
│   ├── api/
│   │   └── app.py              # Flask REST API + WebSocket
│   ├── data/
│   │   └── data_generator.py   # Synthetic data generation
│   ├── models/
│   │   ├── passenger_flow_model.py  # LSTM/GRU prediction model
│   │   └── staff_scheduler_rl.py    # RL staff optimization
│   └── saved_models/           # Trained model storage
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── services/           # API services
│   │   └── App.jsx             # Main application
│   └── package.json
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
├── azure/                      # Azure deployment configs
│   ├── setup-azure.sh          # Automated setup script
│   ├── arm-template.json       # Infrastructure as Code
│   └── AZURE_DEPLOYMENT.md     # Deployment guide
├── .github/
│   └── workflows/
│       └── azure-deploy.yml    # CI/CD pipeline
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (optional)

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data and train model
cd backend/data
python data_generator.py

# Start the API server
cd ../api
python app.py
```

The API will be available at `http://localhost:5000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:5173`

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Frontend: http://localhost
# API: http://localhost:5000
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/current-status` | GET | Current airport status |
| `/api/predict` | POST | Get ML predictions |
| `/api/forecast` | GET | Next 4-hour forecast |
| `/api/alerts` | GET | Active alerts |
| `/api/historical` | GET | Historical data |
| `/api/flights` | GET | Flight schedule |

## 🤖 Machine Learning Models

### LSTM/GRU Time Series Model
- **Input**: 24 hours of historical data (hourly)
- **Output**: 4-hour forecast for security wait, check-in congestion, baggage crowding
- **Architecture**: Stacked LSTM/GRU with dropout and batch normalization
- **Features**: Cyclical time encoding, lag features, rolling statistics

### RL Staff Scheduler (Optional)
- **Algorithm**: PPO (Proximal Policy Optimization)
- **State**: Passenger flow, queue lengths, current staff allocation
- **Action**: Adjust staff ±2 per zone
- **Reward**: Minimize wait times while controlling costs

## 📈 Training the Model

```python
from backend.models.passenger_flow_model import PassengerFlowPredictor
from backend.data.data_generator import AirportDataGenerator

# Generate data
generator = AirportDataGenerator(seed=42)
data = generator.generate_all_data(output_dir='./data', historical_days=365)

# Train model
predictor = PassengerFlowPredictor(
    sequence_length=24,
    forecast_horizon=4,
    model_type='lstm'  # or 'gru', 'bidirectional'
)
predictor.train(data['historical_traffic'], epochs=50)
predictor.save('./saved_models/passenger_flow')
```

## 🎨 Dashboard Features

- **Status Cards**: Real-time passenger count, security wait, check-in wait, baggage status
- **Flow Charts**: 24-hour passenger flow visualization
- **Wait Time Trends**: Multi-zone wait time comparison
- **Forecast View**: Next 4-hour congestion predictions
- **Flight Board**: Upcoming departures with status
- **Alert Panel**: Crowd surge warnings with severity levels
- **Zone Overview**: All airport zones with status indicators

## ☁️ Azure Cloud Deployment

This project includes full **Azure Container Apps** deployment with CI/CD.

### Quick Azure Deployment

```bash
# 1. Login to Azure
az login

# 2. Run automated setup
chmod +x azure/setup-azure.sh
./azure/setup-azure.sh
```

### CI/CD with GitHub Actions

The project includes automated deployment pipeline:

1. **Add GitHub Secrets:**
   - `AZURE_CREDENTIALS` - Azure Service Principal
   - `ACR_USERNAME` - Container Registry username
   - `ACR_PASSWORD` - Container Registry password

2. **Push to main** - Auto-deploys to Azure!

### Azure Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Azure Container Apps Environment            │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │  Frontend App    │ ───▶ │  Backend App     │        │
│  │  (React/Nginx)   │      │  (Flask/ML)      │        │
│  │  External Access │      │  Internal Only   │        │
│  └──────────────────┘      └──────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

📖 See [azure/AZURE_DEPLOYMENT.md](azure/AZURE_DEPLOYMENT.md) for detailed instructions.

## 🔧 Configuration

Environment variables:
```bash
FLASK_ENV=production
MODEL_PATH=/app/backend/saved_models/passenger_flow
DATA_PATH=/app/backend/data/generated_data
```

## 📝 License

MIT License - feel free to use for your airport analytics projects!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
