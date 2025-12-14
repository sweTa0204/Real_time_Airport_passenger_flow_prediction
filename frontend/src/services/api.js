const API_BASE = '/api';

export const apiService = {
  async getCurrentStatus() {
    const res = await fetch(`${API_BASE}/current-status`);
    return res.json();
  },
  async getAlerts() {
    const res = await fetch(`${API_BASE}/alerts`);
    return res.json();
  },
  async getHistoricalData(hours = 24) {
    const res = await fetch(`${API_BASE}/historical?hours=${hours}`);
    return res.json();
  },
  async getForecast(hours = 4) {
    const res = await fetch(`${API_BASE}/forecast?hours=${hours}`);
    return res.json();
  },
  async getFlights(limit = 20) {
    const res = await fetch(`${API_BASE}/flights?limit=${limit}`);
    return res.json();
  },
  async predict(data) {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data })
    });
    return res.json();
  }
};
