import { useId, useState } from 'react'
import { RISK_BANDS, riskBand, scoreText } from '../researchDisplay'
import RiskThresholdTooltip from './RiskThresholdTooltip'
import './ResearchRiskGauge.css'

function Marker({ value, threshold = false }) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const text = `${threshold ? 'Fraud Threshold Line' : 'Risk Score'}: ${scoreText(value)}%`
  return <>
    <button type="button" className={`research-gauge-marker ${threshold ? 'research-gauge-threshold' : 'research-gauge-score'}`}
      style={{ left: `${Math.max(0, Math.min(100, value))}%` }} aria-label={text} aria-describedby={open ? id : undefined}
      onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} onClick={() => setOpen(true)}
      onKeyDown={event => { if (event.key === 'Escape') setOpen(false) }} />
    {open && <span id={id} role="tooltip" className={`research-gauge-tooltip${threshold ? ' research-gauge-tooltip--threshold' : ''}`}>{text}</span>}
  </>
}

export default function ResearchRiskGauge({ score, threshold, compact = false }) {
  return <div className={`research-gauge${compact ? ' research-gauge--compact' : ''}`}>
    <div className="research-gauge-header"><span>Risk Level</span><RiskThresholdTooltip /></div>
    <div className="research-gauge-track" style={{ background: `linear-gradient(to right, ${RISK_BANDS.map((band, i) => `${band.color} ${i * 20}%, ${band.color} ${(i + 1) * 20}%`).join(', ')})` }}>
      {threshold != null && <Marker value={threshold} threshold />}
      <Marker value={score} />
    </div>
    {compact ? <p className="research-gauge-level" style={{ color: riskBand(score).color }}>{riskBand(score).label}</p>
      : <div className="research-gauge-labels">{RISK_BANDS.map(band => <span key={band.label}>{band.label}</span>)}</div>}
  </div>
}
