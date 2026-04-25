import React, { useEffect, useState } from 'react';

const STEPS = [
  'Validating Input',
  'Preprocessing Data',
  'Extracting Features',
  'Running AI Model',
  'Calculating Uncertainty',
  'Finalizing Output',
];

const ProgressPanel = ({ step, setStep, loading }) => {
  const [secondsLeft, setSecondsLeft] = useState(7);

  // Increment step every 1s while loading and step < 6
  useEffect(() => {
    if (!loading) return;
    if (step > 0 && step < STEPS.length) {
      const t = setTimeout(() => setStep(s => s + 1), 1200);
      return () => clearTimeout(t);
    }
  }, [step, loading, setStep]);

  // Countdown timer
  useEffect(() => {
    if (!loading) { setSecondsLeft(7); return; }
    setSecondsLeft(7);
    const interval = setInterval(() => {
      setSecondsLeft(s => (s > 0 ? s - 1 : 0));
    }, 1000);
    return () => clearInterval(interval);
  }, [loading]);

  // Current running step gets a fake percentage
  const getStepPct = (i) => {
    const stepNum = i + 1;
    if (stepNum < step) return null;
    if (stepNum === step) return `${Math.min(100, Math.round(((7 - secondsLeft) / 7) * 100))}%`;
    return '0%';
  };

  const formatTime = (s) => {
    const m = Math.floor(s / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return `00:${m}:${sec}`;
  };

  return (
    <div className="sidebar-section">
      <div className="sidebar-section-title">Progress</div>
      <ul className="progress-list">
        {STEPS.map((name, i) => {
          const stepNum = i + 1;
          const isDone = stepNum < step;
          const isActive = stepNum === step;
          return (
            <li
              key={name}
              className={`progress-item ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}`}
            >
              <div className="progress-item-left">
                <div className="step-dot" />
                {name}
              </div>
              <div>
                {isDone && <span className="step-check">✓</span>}
                {isActive && <span className="step-pct">{getStepPct(i)}</span>}
                {!isDone && !isActive && <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>0%</span>}
              </div>
            </li>
          );
        })}
      </ul>

      {loading && (
        <div className="eta-row">
          <div>Estimated Time Remaining</div>
          <div className="eta-timer">{formatTime(secondsLeft)}</div>
        </div>
      )}
    </div>
  );
};

export default ProgressPanel;
