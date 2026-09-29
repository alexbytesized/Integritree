import { useState, useRef, useEffect } from "react"
import { Info } from "lucide-react"
import "./RiskThresholdTooltip.css"
import { RISK_BANDS, scoreText } from '../researchDisplay'

const RiskThresholdTooltip = ({ position = "bottom", threshold }) => {
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef(null)

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false)
      }
    }
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setIsOpen(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    document.addEventListener("keydown", handleKeyDown)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
      document.removeEventListener("keydown", handleKeyDown)
    }
  }, [])

  return (
    <div
      className="risk-tooltip-wrapper"
      ref={containerRef}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <button
        type="button"
        className="risk-tooltip-trigger"
        onClick={() => setIsOpen(true)}
        aria-label="View risk classification thresholds"
        aria-expanded={isOpen}
      >
        <Info size={16} />
      </button>

      {isOpen && (
        <div className={`risk-tooltip-popover risk-tooltip--${position}`} role="tooltip">
          <div className="risk-tooltip-header">
            <h4 className="risk-tooltip-title">Risk Thresholds</h4>
          </div>
          <div className="risk-tooltip-list">
            {threshold != null && <p>Fraud is predicted at or above {scoreText(threshold)}%.</p>}
            {RISK_BANDS.map(({ label, range, color }) => (
              <div key={label} className="risk-tooltip-item">
                <div className="risk-tooltip-badge-group">
                  <span className="risk-tooltip-dot" style={{ backgroundColor: color }} />
                  <span className="risk-tooltip-label">{label}</span>
                </div>
                <span className="risk-tooltip-range">{range}</span>
              </div>
            ))}
          </div>
          <p className="risk-tooltip-note">Note: The following bands describe model scores, not calibrated real-world probabilities.</p>
        </div>
      )}
    </div>
  )
}

export default RiskThresholdTooltip
