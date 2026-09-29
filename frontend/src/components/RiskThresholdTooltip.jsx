import { useState, useRef, useEffect } from "react"
import { Info } from "lucide-react"
import "./RiskThresholdTooltip.css"

const THRESHOLDS = [
  { label: "Minimal Risk", range: "Below 20", color: "#1DB954" },
  { label: "Low Risk", range: "20 to <40", color: "#F9C923" },
  { label: "Moderate Risk", range: "40 to <60", color: "#F47C20" },
  { label: "High Risk", range: "60 to <80", color: "#D9312B" },
  { label: "Critical Risk", range: "80 to 100", color: "#7A0000" },
]

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
        onClick={() => setIsOpen((prev) => !prev)}
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
            {threshold != null && <p>Fraud is predicted at or above {threshold}%. These bands describe model scores, not calibrated real-world probabilities.</p>}
            {THRESHOLDS.map(({ label, range, color }) => (
              <div key={label} className="risk-tooltip-item">
                <div className="risk-tooltip-badge-group">
                  <span className="risk-tooltip-dot" style={{ backgroundColor: color }} />
                  <span className="risk-tooltip-label">{label}</span>
                </div>
                <span className="risk-tooltip-range">{range}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default RiskThresholdTooltip
