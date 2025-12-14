import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, Legend } from 'recharts';

export default function Dashboard({ status, historicalData, forecast, flights }) {
  const formatTime = (ts) => new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className="space-y-6">
      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatusCard title="Total Passengers" value={status?.total_passengers || 0} icon="👥" color="blue" subtitle="Current hour" />
        <StatusCard title="Security Wait" value={`${status?.security?.current_wait || 0} min`} icon="🔒" 
          color={status?.security?.status === 'critical' ? 'red' : status?.security?.status === 'busy' ? 'yellow' : 'green'}
          subtitle={`${status?.security?.gates_open || 0} gates open`} />
        <StatusCard title="Check-in Wait" value={`${status?.checkin?.current_wait || 0} min`} icon="📋"
          color={status?.checkin?.status === 'critical' ? 'red' : status?.checkin?.status === 'busy' ? 'yellow' : 'green'}
          subtitle={`${status?.checkin?.counters_open || 0} counters`} />
        <StatusCard title="Baggage Claim" value={`${status?.baggage?.passengers_waiting || 0}`} icon="🧳"
          color={status?.baggage?.status === 'critical' ? 'red' : status?.baggage?.status === 'busy' ? 'yellow' : 'green'}
          subtitle="passengers waiting" />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Historical Passenger Flow */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">📊 Passenger Flow (24h)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={historicalData}>
              <defs>
                <linearGradient id="colorPass" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="timestamp" tickFormatter={formatTime} stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                labelFormatter={formatTime} />
              <Area type="monotone" dataKey="passengers" stroke="#06b6d4" fillOpacity={1} fill="url(#colorPass)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Wait Times */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">⏱️ Wait Times (24h)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="timestamp" tickFormatter={formatTime} stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                labelFormatter={formatTime} />
              <Legend />
              <Line type="monotone" dataKey="security_wait" name="Security" stroke="#f97316" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="checkin_wait" name="Check-in" stroke="#22c55e" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="baggage_wait" name="Baggage" stroke="#8b5cf6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Forecast & Flights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Forecast */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">🔮 Predicted Congestion (Next 4h)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={forecast}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="timestamp" tickFormatter={formatTime} stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                labelFormatter={formatTime} />
              <Legend />
              <Bar dataKey="security_wait" name="Security (min)" fill="#f97316" radius={[4, 4, 0, 0]} />
              <Bar dataKey="checkin_congestion" name="Check-in (min)" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Upcoming Flights */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">✈️ Upcoming Departures</h3>
          <div className="space-y-3 max-h-[200px] overflow-y-auto">
            {flights?.slice(0, 5).map((flight, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center text-blue-400 font-bold text-sm">
                    {flight.flight_id?.slice(0, 2)}
                  </div>
                  <div>
                    <p className="font-medium text-white">{flight.flight_id}</p>
                    <p className="text-sm text-slate-400">{flight.destination} • Gate {flight.gate}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-medium text-white">{formatTime(flight.departure_time)}</p>
                  <p className={`text-sm ${flight.status === 'Delayed' ? 'text-red-400' : 'text-green-400'}`}>{flight.status}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Zone Status */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4">🏢 Zone Status Overview</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <ZoneCard zone="Security" icon="🔒" level={status?.security?.status || 'normal'} value={`${status?.security?.current_wait || 0}m wait`} />
          <ZoneCard zone="Check-in" icon="📋" level={status?.checkin?.status || 'normal'} value={`${status?.checkin?.counters_open || 0} counters`} />
          <ZoneCard zone="Baggage" icon="🧳" level={status?.baggage?.status || 'normal'} value={`${status?.baggage?.active_carousels || 0} active`} />
          <ZoneCard zone="Retail" icon="🛍️" level="normal" value="Open" />
          <ZoneCard zone="Lounges" icon="🛋️" level="normal" value="Available" />
        </div>
      </div>
    </div>
  );
}

function StatusCard({ title, value, icon, color, subtitle }) {
  const colors = {
    blue: 'from-blue-500/20 to-blue-600/5 border-blue-500/30',
    green: 'from-green-500/20 to-green-600/5 border-green-500/30',
    yellow: 'from-yellow-500/20 to-yellow-600/5 border-yellow-500/30',
    red: 'from-red-500/20 to-red-600/5 border-red-500/30'
  };
  return (
    <div className={`bg-gradient-to-br ${colors[color]} border rounded-xl p-5`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-slate-400 text-sm">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
        </div>
        <span className="text-2xl">{icon}</span>
      </div>
    </div>
  );
}

function ZoneCard({ zone, icon, level, value }) {
  const levelColors = { normal: 'bg-green-500', busy: 'bg-yellow-500', critical: 'bg-red-500' };
  return (
    <div className="bg-slate-700/50 rounded-lg p-4 text-center">
      <span className="text-2xl">{icon}</span>
      <p className="font-medium text-white mt-2">{zone}</p>
      <p className="text-xs text-slate-400">{value}</p>
      <div className={`w-3 h-3 ${levelColors[level]} rounded-full mx-auto mt-2`}></div>
    </div>
  );
}
