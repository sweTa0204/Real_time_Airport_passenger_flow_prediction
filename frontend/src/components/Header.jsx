import React from 'react';

export default function Header({ connected }) {
  return (
    <header className="bg-slate-800 border-b border-slate-700 px-6 py-4">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-3xl">✈️</span>
          <div>
            <h1 className="text-xl font-bold text-white">Airport Flow Analytics</h1>
            <p className="text-sm text-slate-400">Real-Time Passenger Flow Prediction</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
            <span className="text-sm text-slate-400">{connected ? 'Live' : 'Offline'}</span>
          </div>
          <div className="text-sm text-slate-400">
            {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>
    </header>
  );
}
