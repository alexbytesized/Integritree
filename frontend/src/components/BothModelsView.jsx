import "./BothModelsView.css"

const RISK_LABELS = ["Minimal Risk", "Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]

const getRiskLevel = (score) => {
  if (score < 20) return "Minimal Risk"
  if (score < 40) return "Low Risk"
  if (score < 60) return "Moderate Risk"
  if (score < 80) return "High Risk"
  return "Critical Risk"
}

/* Compact inline gauge used inside the side-by-side cards */
const MiniGauge = ({ score }) => (
  <div className="bmv-gauge-wrapper">
    <div className="bmv-gauge-track">
      <div
        className="bmv-gauge-thumb"
        style={{ left: `clamp(1%, ${score}%, 99%)` }}
        aria-label={`Risk score ${score}/100`}
      />
    </div>
    <p className="bmv-gauge-label">{getRiskLevel(score)}</p>
  </div>
)

/* One stacked info card: blue pill header + white body */
const StackedCard = ({ header, children }) => (
  <div className="bmv-stacked-card">
    <div className="bmv-stacked-card-header">{header}</div>
    <div className="bmv-stacked-card-body">{children}</div>
  </div>
)

/* The left/right compact model summary */
const ModelColumn = ({ label, prediction, riskScore, groundTruth, outcome }) => {
  const isFraudPred = prediction  === "Fraudulent" || prediction  === "Fraud"
  const isFraudGT   = groundTruth === "Fraudulent" || groundTruth === "Fraud"
  return (
    <div className="bmv-model-col">
      <div className="bmv-model-col-header">{label}</div>
      <div className="bmv-model-col-body">
        <MiniGauge score={riskScore} />
        <StackedCard header="Prediction">
          <span className={isFraudPred ? "bmv-fraud" : "bmv-legit"}>{prediction}</span>
        </StackedCard>
        <StackedCard header="Risk Score">
          <span className="bmv-value"><strong>{riskScore}</strong> out of <strong>100</strong></span>
        </StackedCard>
        <StackedCard header="Ground Truth">
          <span className={isFraudGT ? "bmv-fraud" : "bmv-legit"}>{groundTruth}</span>
        </StackedCard>
        <StackedCard header="Outcome">
          <span className="bmv-value bmv-value--bold">{outcome}</span>
        </StackedCard>
      </div>
    </div>
  )
}

/* Full-width expanded card with interpretation + SHAP */
const ExpandedCard = ({ label, riskScore, prediction, interpretation, shapSummary, shapLink }) => {
  const riskLevel   = getRiskLevel(riskScore)
  const isFraudPred = prediction === "Fraudulent" || prediction === "Fraud"

  const defaultInterpretation = (
    <>
      Your transaction is at{" "}
      <span className="bmv-highlight-risk">{riskLevel}</span>{" "}
      of being fraudulent and was classified as potentially fraudulent.
    </>
  )

  const defaultShap =
    `The transaction was classified as ${prediction} primarily because of its ` +
    `high transaction amount, CASH_OUT transaction type, large decrease in the sender's balance, ` +
    `and unusual change in the recipient's balance. These factors increased the model's fraud ` +
    `prediction, resulting in a ${riskLevel} score of ${riskScore}/100.`

  return (
    <div className="bmv-expanded-card">
      <div className="bmv-expanded-header">{label}</div>
      <div className="bmv-expanded-body">
        <h4 className="bmv-section-heading">INTERPRETATION</h4>
        <p className="bmv-section-text">{interpretation ?? defaultInterpretation}</p>

        <h4 className="bmv-section-heading">SHAP EXPLANATION</h4>
        <p className="bmv-section-text">{shapSummary ?? defaultShap}</p>

        <div className="bmv-shap-link-row">
          {shapLink
            ? <a href={shapLink} className="bmv-shap-link">See Full SHAP Evaluation →</a>
            : <span className="bmv-shap-link bmv-shap-link--placeholder">See Full SHAP Evaluation →</span>
          }
        </div>
      </div>
    </div>
  )
}

const BothModelsView = ({ rfSmote, benchmark }) => (
  <div className="bmv-root">
    {/* Side-by-side compact model summary */}
    <div className="bmv-columns">
      <ModelColumn
        label="RF-SMOTE"
        prediction={rfSmote.prediction}
        riskScore={rfSmote.riskScore}
        groundTruth={rfSmote.groundTruth}
        outcome={rfSmote.outcome}
      />
      <ModelColumn
        label="Benchmark RF"
        prediction={benchmark.prediction}
        riskScore={benchmark.riskScore}
        groundTruth={benchmark.groundTruth}
        outcome={benchmark.outcome}
      />
    </div>

    {/* Full-width RF-SMOTE interpretation + SHAP */}
    <ExpandedCard
      label="RF-SMOTE"
      riskScore={rfSmote.riskScore}
      prediction={rfSmote.prediction}
      interpretation={rfSmote.interpretation}
      shapSummary={rfSmote.shapSummary}
      shapLink={rfSmote.shapLink}
    />

    {/* Full-width Benchmark RF interpretation + SHAP */}
    <ExpandedCard
      label="Benchmark RF"
      riskScore={benchmark.riskScore}
      prediction={benchmark.prediction}
      interpretation={benchmark.interpretation}
      shapSummary={benchmark.shapSummary}
      shapLink={benchmark.shapLink}
    />
  </div>
)

export default BothModelsView
