import "./ModelResultView.css"
import RiskThresholdTooltip from "./RiskThresholdTooltip"

const RISK_LABELS = ["Minimal Risk", "Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]

const getRiskLevel = (score) => {
  if (score < 20) return "Minimal Risk"
  if (score < 40) return "Low Risk"
  if (score < 60) return "Moderate Risk"
  if (score < 80) return "High Risk"
  return "Critical Risk"
}

const RiskGauge = ({ score, threshold }) => (
  <div className="mrv-gauge-wrapper">
    <div className="mrv-gauge-header">
      <span className="mrv-gauge-title">Risk Level</span>
      <RiskThresholdTooltip position="bottom" threshold={threshold} />
    </div>
    <div className="mrv-gauge-track">
      {threshold != null && <span title={`Fraud threshold ${threshold}%`} style={{ position: "absolute", left: `${threshold}%`, height: "100%", borderLeft: "2px solid black" }} />}
      <div
        className="mrv-gauge-thumb"
        style={{ left: `clamp(1%, ${score}%, 99%)` }}
        role="img"
        aria-label={`Risk score ${score} out of 100`}
      />
    </div>
    <div className="mrv-gauge-labels">
      {RISK_LABELS.map((label, i) => (
        <span key={label} className="mrv-gauge-label" style={{ left: `${i * 25}%` }}>
          {label}
        </span>
      ))}
    </div>
  </div>
)

const InfoCard = ({ header, full, children }) => (
  <div className={`mrv-info-card${full ? " mrv-info-card--full" : ""}`}>
    <div className="mrv-info-card-header">{header}</div>
    <div className="mrv-info-card-body">{children}</div>
  </div>
)

/**
 * ModelResultView
 * Props:
 *   prediction   – "Fraudulent" | "Legitimate"
 *   riskScore    – 0–100
 *   groundTruth  – "Fraudulent" | "Legitimate"
 *   outcome      – e.g. "True Positive"
 *   interpretation – string or JSX (optional, defaults to generated text)
 *   shapSummary  – string (optional, defaults to generated text)
 *   shapLink     – href string for the "See Full SHAP Evaluation" link (optional)
 */
const ModelResultView = ({
  prediction = "Unavailable",
  riskScore = 0,
  threshold,
  groundTruth = "Unavailable",
  outcome = "Unavailable",
  interpretation,
  shapSummary,
  shapLink,
}) => {
  const riskLevel   = getRiskLevel(riskScore)
  const isFraudPred = prediction  === "Fraudulent"
  const isFraudGT   = groundTruth === "Fraudulent"

  const defaultInterpretation = (
    <>
      Your transaction is at{" "}
      <span className="mrv-highlight-risk">{riskLevel}</span>{" "}
      of being fraudulent and was predicted as{" "}
      <span className={isFraudPred ? "mrv-highlight-fraud" : "mrv-highlight-legit"}>
        {prediction}
      </span>.
    </>
  )

  const defaultShap = 'No explanation has been computed for this record.'


  return (
    <div className="mrv-root">
      {/* ── Risk gauge ── */}
      <RiskGauge score={riskScore} threshold={threshold} />

      {/* ── 2×2 cards ── */}
      <div className="mrv-cards-grid">
        <InfoCard header="Prediction">
          <span className={isFraudPred ? "mrv-highlight-fraud" : "mrv-highlight-legit"}>
            {prediction}
          </span>
        </InfoCard>

        <InfoCard header="Risk Score">
          <span className="mrv-risk-score">
            <strong>{riskScore}</strong> out of <strong>100</strong>
          </span>
        </InfoCard>

        <InfoCard header="Ground Truth">
          <span className={isFraudGT ? "mrv-highlight-fraud" : "mrv-highlight-legit"}>
            {groundTruth}
          </span>
        </InfoCard>

        <InfoCard header="Outcome">
          <span className="mrv-outcome">{outcome}</span>
        </InfoCard>
      </div>

      {/* ── Interpretation ── */}
      <InfoCard header="Interpretation" full>
        <p className="mrv-text">{interpretation ?? defaultInterpretation}</p>
      </InfoCard>

      {/* ── SHAP Explanation ── */}
      <InfoCard header="SHAP Explanation" full>
        <p className="mrv-text">{shapSummary ?? defaultShap}</p>
        <div className="mrv-shap-link-row">
          {shapLink
            ? <a href={shapLink} className="mrv-shap-link">See Full SHAP Evaluation →</a>
            : <span className="mrv-shap-link mrv-shap-link--placeholder">See Full SHAP Evaluation →</span>
          }
        </div>
      </InfoCard>
    </div>
  )
}

export default ModelResultView

