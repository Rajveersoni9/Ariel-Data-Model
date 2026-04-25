import React from 'react';
import { PLANET_NAMES } from './PlanetCarousel';

const MOCK_RESULTS = [
  { id: 'HD-982547 b', rmse: '0.1287', gll: '0.0843', highlight: true },
  { id: 'TOI-1781 c', rmse: '0.1432', gll: '0.0931' },
  { id: 'K2-18 b', rmse: '0.1569', gll: '0.1017' },
  { id: 'EPIC 249893012 b', rmse: '0.1673', gll: '0.1124' },
  { id: 'Kepler-1649 c', rmse: '0.1789', gll: '0.1218' },
];

const OutputTable = ({ spectrum, sigma }) => {
  const handleDownload = () => {
    if (!spectrum || !sigma) return;
    const rows = spectrum.map((v, i) => `${i},${v.toFixed(6)},${sigma[i].toFixed(6)}`).join('\n');
    const csv = `Index,Spectrum,Sigma\n${rows}`;
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ariel_prediction_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="result-summary fade-up">
      <div className="panel-title">Result Summary</div>
      <div className="result-table-wrap">
        <table className="result-table">
          <thead>
            <tr>
              <th>Planet ID</th>
              <th>RMSE</th>
              <th>Scaled GLL</th>
            </tr>
          </thead>
          <tbody>
            {MOCK_RESULTS.map((row) => (
              <tr key={row.id} className={row.highlight ? 'highlighted' : ''}>
                <td>
                  <div className="planet-id-cell">
                    {row.highlight && <div className="planet-dot" />}
                    {row.id}
                  </div>
                </td>
                <td>{row.rmse}</td>
                <td>{row.gll}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button className="download-btn" onClick={handleDownload}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Download Results
      </button>
    </div>
  );
};

export default OutputTable;
