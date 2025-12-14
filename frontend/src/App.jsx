import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import Dashboard from './components/Dashboard';
import AlertPanel from './components/AlertPanel';
import Header from './components/Header';
import { apiService } from './services/api';

function App() {
  const [status, setStatus] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [historicalData, setHistoricalData] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [flights, setFlights] = useState([]);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [statusRes, alertsRes, histRes, forecastRes, flightsRes] = await Promise.all([
          apiService.getCurrentStatus(),
          apiService.getAlerts(),
          apiService.getHistoricalData(24),
          apiService.getForecast(4),
          apiService.getFlights(10)
        ]);
        setStatus(statusRes);
        setAlerts(alertsRes.alerts || []);
        setHistoricalData(histRes.data || []);
        setForecast(forecastRes.forecast || []);
        setFlights(flightsRes.flights || []);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching initial data:', error);
        // Use mock data on error
        setStatus(getMockStatus());
        setAlerts(getMockAlerts());
        setHistoricalData(getMockHistorical());
        setForecast(getMockForecast());
        setFlights(getMockFlights());
        setLoading(false);
      }
    };
    fetchInitialData();

    // WebSocket - connect via nginx proxy
    const socket = io('/', { transports: ['websocket', 'polling'] });
    socket.on('connect', () => { setConnected(true); socket.emit('subscribe_updates', {}); });
    socket.on('disconnect', () => setConnected(false));
    socket.on('status_update', (data) => {
      setStatus(prev => ({ ...prev, timestamp: data.timestamp, total_passengers: data.passengers,
        security: { ...prev?.security, current_wait: data.security_wait },
        checkin: { ...prev?.checkin, current_wait: data.checkin_wait },
        baggage: { ...prev?.baggage, avg_wait: data.baggage_wait } }));
      setHistoricalData(prev => [...prev, { timestamp: data.timestamp, passengers: data.passengers,
        security_wait: data.security_wait, checkin_wait: data.checkin_wait, baggage_wait: data.baggage_wait }].slice(-48));
    });
    socket.on('alert', (alert) => setAlerts(prev => [{ ...alert, id: `rt-${Date.now()}` }, ...prev.slice(0, 9)]));
    const refreshInterval = setInterval(async () => {
      try { const [f, a] = await Promise.all([apiService.getForecast(4), apiService.getAlerts()]);
        setForecast(f.forecast || []); setAlerts(a.alerts || []);
      } catch (e) { console.error(e); }
    }, 30000);
    return () => { socket.disconnect(); clearInterval(refreshInterval); };
  }, []);

  const dismissAlert = (alertId) => setAlerts(prev => prev.filter(a => a.id !== alertId));

  if (loading) return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-cyan-400 mx-auto mb-4"></div>
        <p className="text-slate-400 text-lg">Loading Airport Dashboard...</p>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-900">
      <Header connected={connected} />
      <main className="container mx-auto px-4 py-6">
        {alerts.length > 0 && alerts[0].severity === 'high' && (
          <div className="mb-6 bg-red-900/50 border border-red-500 rounded-lg p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-2xl">⚠️</span>
              <div>
                <p className="font-semibold text-red-200">{alerts[0].message}</p>
                <p className="text-sm text-red-300">Expected in {alerts[0].minutes_ahead} minutes</p>
              </div>
            </div>
            <button onClick={() => dismissAlert(alerts[0].id)} className="text-red-300 hover:text-white">✕</button>
          </div>
        )}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-3">
            <Dashboard status={status} historicalData={historicalData} forecast={forecast} flights={flights} />
          </div>
          <div className="lg:col-span-1">
            <AlertPanel alerts={alerts} onDismiss={dismissAlert} />
          </div>
        </div>
      </main>
    </div>
  );
}

// Mock data helpers
const getMockStatus = () => ({ timestamp: new Date().toISOString(), total_passengers: 450,
  security: { current_wait: 15.2, queue_length: 45, throughput: 150, gates_open: 4, status: 'normal' },
  checkin: { current_wait: 8.5, counters_open: 15, kiosks_active: 22, status: 'normal' },
  baggage: { active_carousels: 4, passengers_waiting: 65, avg_wait: 18.3, status: 'normal' },
  flights: { departures_next_hour: 12, arrivals_next_hour: 10, delays: 2, cancellations: 0 } });
const getMockAlerts = () => [{ id: '1', type: 'crowd_surge', area: 'Security', severity: 'medium',
  message: 'Expected security queue increase in 25 minutes', minutes_ahead: 25, timestamp: new Date().toISOString() }];
const getMockHistorical = () => Array.from({length: 24}, (_, i) => ({ timestamp: new Date(Date.now() - (24-i)*3600000).toISOString(),
  passengers: 200 + Math.random()*300, security_wait: 10 + Math.random()*10, checkin_wait: 5 + Math.random()*8, baggage_wait: 12 + Math.random()*10 }));
const getMockForecast = () => Array.from({length: 4}, (_, i) => ({ timestamp: new Date(Date.now() + (i+1)*3600000).toISOString(),
  security_wait: 12 + Math.random()*10, checkin_congestion: 6 + Math.random()*8, baggage_crowding: 150 + Math.random()*200 }));
const getMockFlights = () => ['Delta', 'United', 'American', 'Southwest', 'JetBlue'].map((airline, i) => ({
  flight_id: `${airline.slice(0,2).toUpperCase()}${1000+i}`, airline, destination: ['LAX','JFK','ORD','DFW','MIA'][i],
  departure_time: new Date(Date.now() + (i+1)*30*60000).toISOString(), gate: `G${10+i}`, terminal: `T${(i%3)+1}`, status: i===2?'Delayed':'On Time', expected_passengers: 150+i*20 }));

export default App;
