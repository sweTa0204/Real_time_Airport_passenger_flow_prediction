import React from 'react';

export default function AlertPanel({ alerts, onDismiss }) {
  const severityStyles = {
    high: 'bg-red-900/30 border-red-500/50 text-red-200',
    medium: 'bg-yellow-900/30 border-yellow-500/50 text-yellow-200',
    low: 'bg-blue-900/30 border-blue-500/50 text-blue-200'
  };
  const severityIcons = { high: '🚨', medium: '⚠️', low: 'ℹ️' };

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">🔔 Alerts</h3>
        <span className="bg-red-500/20 text-red-400 text-xs px-2 py-1 rounded-full">
          {alerts.length} Active
        </span>
      </div>
      <div className="space-y-3 max-h-[500px] overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="text-center py-8 text-slate-500">
            <span className="text-3xl">✅</span>
            <p className="mt-2">No active alerts</p>
          </div>
        ) : (
          alerts.map((alert) => (
            <div key={alert.id} className={`p-4 rounded-lg border ${severityStyles[alert.severity || 'medium']}`}>
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-2">
                  <span className="text-lg">{severityIcons[alert.severity || 'medium']}</span>
                  <div>
                    <p className="font-medium text-sm">{alert.message}</p>
                    <p className="text-xs opacity-75 mt-1">
                      {alert.area} • In {alert.minutes_ahead} min
                    </p>
                  </div>
                </div>
                <button onClick={() => onDismiss(alert.id)} className="text-slate-400 hover:text-white text-sm">✕</button>
              </div>
            </div>
          ))
        )}
      </div>
      <div className="mt-4 pt-4 border-t border-slate-700">
        <h4 className="text-sm font-medium text-slate-400 mb-2">Alert Thresholds</h4>
        <div className="text-xs text-slate-500 space-y-1">
          <p>• Security: &gt;25 min wait</p>
          <p>• Check-in: &gt;15 min wait</p>
          <p>• Baggage: &gt;500 passengers</p>
        </div>
      </div>
    </div>
  );
}
