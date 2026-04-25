import React, { useState, useRef, useCallback } from 'react';
import apiService from '../services/api';
import EarthViewer from '../components/EarthViewer';
import PlanetCarousel, { PLANET_NAMES } from '../components/PlanetCarousel';
import ProgressPanel from '../components/ProgressPanel';
import PredictionGraph from '../components/PredictionGraph';
import SigmaGraph from '../components/SigmaGraph';
import OutputTable from '../components/OutputTable';

const Dashboard = () => {
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0);
  const [spectrum, setSpectrum] = useState(null);
  const [sigma, setSigma] = useState(null);
  const [error, setError] = useState(null);
  const [rs, setRs] = useState('1.23');
  const [inclination, setInclination] = useState('87.45');
  const [fileName, setFileName] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [exoPlanetIndex, setExoPlanetIndex] = useState(0);
  const fileInputRef = useRef(null);

  const handleFileChange = (file) => {
    if (file) setFileName(file.name);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileChange(file);
  };

  const handleStart = async () => {
    setLoading(true);
    setStep(1);
    setSpectrum(null);
    setSigma(null);
    setError(null);

    // 3D array: 1 planet x 200 time steps x 283 wavelengths
    const mockData = [Array.from({ length: 200 }, () => Array(283).fill(0.5))];
    const mockStarInfo = [parseFloat(rs) || 1.0, parseFloat(inclination) || 87.0];

    try {
      const response = await apiService.predict(mockData, mockStarInfo);
      setLoading(false);
      setStep(6);
      setSpectrum(response.spectrum);
      setSigma(response.sigma);
    } catch (err) {
      setLoading(false);
      setError('Connection failed. Is the backend running?');
      setStep(0);
    }
  };

  const handleIndexChange = useCallback((val) => {
    setExoPlanetIndex(typeof val === 'function' ? val : val);
  }, []);

  return (
    <div className="app-wrapper">
      {/* ===== HEADER ===== */}
      <header className="top-header">
        <div className="header-logo">
          <div className="logo-icon">✦</div>
          <div className="logo-text">
            ARIEL
            <span>DATA CHALLENGE</span>
          </div>
        </div>

        <div className="header-title">
          <h1>Exoplanet Spectrum Predictor</h1>
          <p className="subtitle">
            AI Powered <span>·</span> Real Time Analysis
          </p>
        </div>

        <div className="api-status">
          API STATUS
          <div className="dot" />
          <span className="status-text">CONNECTED</span>
        </div>
      </header>

      {/* ===== MAIN ===== */}
      <div className="main-content">

        {/* === LEFT SIDEBAR === */}
        <aside className="left-sidebar">
          {/* Input Controls */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Input Controls</div>

            <label className="upload-label">Upload Calibrated Data</label>
            <div
              className={`drop-zone ${isDragOver ? 'active' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".npy"
                style={{ display: 'none' }}
                onChange={(e) => handleFileChange(e.target.files[0])}
              />
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              <p>
                Drag & drop .npy file here<br />
                or <span className="browse-link">browse file</span>
              </p>
            </div>
            {fileName && <div className="file-name-display">📄 {fileName}</div>}
          </div>

          {/* Star Info */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Star Information</div>
            <div className="form-row">
              <label>Rs (Solar Radii)</label>
              <input
                type="number"
                value={rs}
                onChange={(e) => setRs(e.target.value)}
                step="0.01"
              />
            </div>
            <div className="form-row">
              <label>Inclination (deg)</label>
              <input
                type="number"
                value={inclination}
                onChange={(e) => setInclination(e.target.value)}
                step="0.01"
              />
            </div>

            {error && (
              <div style={{ fontSize: 10, color: '#ef4444', marginTop: 8, lineHeight: 1.4 }}>
                ⚠ {error}
              </div>
            )}

            <button className="start-btn" onClick={handleStart} disabled={loading}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3"/>
              </svg>
              {loading ? 'Analyzing...' : 'Start Analysis'}
            </button>
          </div>

          {/* Progress */}
          <ProgressPanel step={step} setStep={setStep} loading={loading} />
        </aside>

        {/* === CENTER STAGE === */}
        <div className="center-stage">
          {/* Planet Visualizations */}
          <div className="planet-display">
            <span className="comparison-label">Comparison Mode</span>

            {/* Earth */}
            <div className="planet-slot">
              <div className="planet-slot-label">
                <h3>EARTH</h3>
                <p>Reference Planet</p>
              </div>
              <div className="planet-canvas-wrap">
                <EarthViewer />
              </div>
            </div>

            <div className="vs-divider">VS</div>

            {/* Exoplanet */}
            <div className="planet-slot">
              <div className="planet-slot-label">
                <h3>TARGET EXOPLANET</h3>
                <p>{PLANET_NAMES[exoPlanetIndex]}</p>
              </div>
              <div className="planet-canvas-wrap" style={{ display: 'flex' }}>
                <PlanetCarousel
                  loading={loading}
                  activeIndex={exoPlanetIndex}
                  onIndexChange={handleIndexChange}
                />
              </div>
            </div>
          </div>

          {/* Data Row */}
          {spectrum && sigma && (
            <div className="data-row fade-up">
              {/* Charts */}
              <div className="analysis-output">
                <div className="panel-title">Analysis Output</div>
                <div className="graphs-row">
                  <PredictionGraph spectrum={spectrum} />
                  <SigmaGraph sigma={sigma} />
                </div>
              </div>

              {/* Result Summary */}
              <OutputTable spectrum={spectrum} sigma={sigma} />
            </div>
          )}
        </div>
      </div>

      {/* ===== FOOTER ===== */}
      <footer className="bottom-footer">
        <div className="footer-icons">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
          </svg>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
          </svg>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/>
          </svg>
        </div>

        <div className="footer-text">
          Ariel Data Challenge 2025 · Predicting Exoplanet Atmospheres
        </div>

        <div className="help-icon">?</div>
      </footer>
    </div>
  );
};

export default Dashboard;
