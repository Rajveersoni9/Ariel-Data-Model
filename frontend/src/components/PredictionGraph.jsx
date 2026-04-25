import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: 'rgba(5,8,20,0.95)',
        border: '1px solid #00d4ff',
        borderRadius: 4,
        padding: '6px 10px',
        fontSize: 10,
        color: '#00d4ff',
      }}>
        <div style={{ color: '#64748b', marginBottom: 2 }}>λ: {label}</div>
        <div>{Number(payload[0].value).toFixed(4)}</div>
      </div>
    );
  }
  return null;
};

const PredictionGraph = ({ spectrum }) => {
  const data = spectrum.map((val, i) => ({
    wavelength: (0.5 + (i / spectrum.length) * 4.5).toFixed(1),
    value: val,
  }));

  return (
    <div className="graph-box fade-up">
      <div className="graph-box-header">
        <div className="graph-box-title cyan">Predicted Spectrum</div>
        <svg className="expand-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="15 3 21 3 21 9" />
          <polyline points="9 21 3 21 3 15" />
          <line x1="21" y1="3" x2="14" y2="10" />
          <line x1="3" y1="21" x2="10" y2="14" />
        </svg>
      </div>
      <div className="chart-area">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 5, left: -30, bottom: 14 }}>
            <defs>
              <linearGradient id="spectrumGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,37,64,0.6)" />
            <XAxis
              dataKey="wavelength"
              stroke="#334155"
              tick={{ fontSize: 7, fill: '#64748b' }}
              label={{ value: 'WAVELENGTH (μm)', position: 'insideBottom', offset: -6, style: { fontSize: 7, fill: '#64748b' } }}
            />
            <YAxis
              stroke="#334155"
              tick={{ fontSize: 7, fill: '#64748b' }}
              label={{ value: 'FLUX', angle: -90, position: 'insideLeft', offset: 10, style: { fontSize: 7, fill: '#64748b' } }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="value" stroke="#00d4ff" strokeWidth={1.5} fill="url(#spectrumGrad)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PredictionGraph;
