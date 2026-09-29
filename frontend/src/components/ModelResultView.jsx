import "./ModelResultView.css"
import ResearchRiskGauge from './ResearchRiskGauge'
import ModelInterpretation from './ModelInterpretation'
import { scoreText } from '../researchDisplay'

const InfoCard = ({ header, full, children }) => (
  <div className={`mrv-info-card${full ? " mrv-info-card--full" : ""}`}>
    <div className="mrv-info-card-header">{header}</div>
    <div className="mrv-info-card-body">{children}</div>
  </div>
)

const ModelResultView = ({
  prediction = "Unavailable",
  riskScore = 0,
  threshold,
  groundTruth = "Unavailable",
  outcome = "Unavailable",
  modelName = 'RF-SMOTE',
  shapSummary,
  onOpenShap,
}) => {
  const isFraudPred = prediction  === "Fraudulent"
  const isFraudGT   = groundTruth === "Fraudulent"

  const defaultShap = 'No explanation has been computed for this record.'


  return (
    <div className="mrv-root">
      {/* ── Risk gauge ── */}
      <ResearchRiskGauge score={riskScore} threshold={threshold} />

      {/* ── 2×2 cards ── */}
      <div className="mrv-cards-grid">
        <InfoCard header="Prediction">
          <span className={isFraudPred ? "mrv-highlight-fraud" : "mrv-highlight-legit"}>
            {prediction}
          </span>
        </InfoCard>

        <InfoCard header="Risk Score">
          <span className="mrv-risk-score">
            <strong>{scoreText(riskScore)}</strong> out of <strong>100</strong>
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
        <p className="mrv-text"><ModelInterpretation modelName={modelName} prediction={prediction} riskScore={riskScore} /></p>
      </InfoCard>

      {/* ── SHAP Explanation ── */}
      <InfoCard header="SHAP Explanation" full>
        <p className="mrv-text">{shapSummary ?? defaultShap}</p>
        <div className="mrv-shap-link-row">
          <button type="button" disabled={!onOpenShap} onClick={onOpenShap} className="mrv-shap-link">See Full SHAP Evaluation &rarr;</button>
        </div>
      </InfoCard>
    </div>
  )
}

export default ModelResultView

