import "./BothModelsView.css"


import ResearchRiskGauge from './ResearchRiskGauge'
import ModelInterpretation from './ModelInterpretation'
import { scoreText } from '../researchDisplay'

/* One stacked info card: blue pill header + white body */
const StackedCard = ({ header, children }) => (
  <div className="bmv-stacked-card">
    <div className="bmv-stacked-card-header">{header}</div>
    <div className="bmv-stacked-card-body">{children}</div>
  </div>
)

/* The left/right compact model summary */
const ModelColumn = ({ label, prediction, riskScore, groundTruth, outcome, threshold, showGroundTruth }) => {
  const isFraudPred = prediction  === "Fraudulent" || prediction  === "Fraud"
  const isFraudGT   = groundTruth === "Fraudulent" || groundTruth === "Fraud"
  return (
    <div className="bmv-model-col">
      <div className="bmv-model-col-header">{label}</div>
      <div className="bmv-model-col-body">
        <ResearchRiskGauge compact score={riskScore} threshold={threshold} />
        <StackedCard header="Prediction">
          <span className={isFraudPred ? "bmv-fraud" : "bmv-legit"}>{prediction}</span>
        </StackedCard>
        <StackedCard header="Risk Score">
          <span className="bmv-value"><strong>{scoreText(riskScore)}</strong> out of <strong>100</strong></span>
        </StackedCard>
        {showGroundTruth && <><StackedCard header="Ground Truth">
          <span className={isFraudGT ? "bmv-fraud" : "bmv-legit"}>{groundTruth}</span>
        </StackedCard>
        <StackedCard header="Outcome">
          <span className="bmv-value bmv-value--bold">{outcome}</span>
        </StackedCard></>}
      </div>
    </div>
  )
}

/* Full-width expanded card with interpretation + SHAP */
const ExpandedCard = ({ modelName, riskScore, prediction, shapSummary, onOpenShap }) => {
  const defaultShap = 'No explanation has been computed for this record.'


  return (
    <div className="bmv-expanded-card">
      <div className="bmv-expanded-header">{modelName}</div>
      <div className="bmv-expanded-body">
        <h4 className="bmv-section-heading">INTERPRETATION</h4>
        <p className="bmv-section-text"><ModelInterpretation modelName={modelName} prediction={prediction} riskScore={riskScore} /></p>

        <h4 className="bmv-section-heading">SHAP EXPLANATION</h4>
        <p className="bmv-section-text">{shapSummary ?? defaultShap}</p>

        <div className="bmv-shap-link-row">
          <button type="button" disabled={!onOpenShap} onClick={onOpenShap} className="bmv-shap-link">See Full SHAP Evaluation &rarr;</button>
        </div>
      </div>
    </div>
  )
}

const BothModelsView = ({ rfSmote, benchmark, showGroundTruth = true }) => (
  <div className="bmv-root">
    {/* Side-by-side compact model summary */}
    <div className="bmv-columns">
      <ModelColumn
        label="RF-SMOTE"
        showGroundTruth={showGroundTruth}
        prediction={rfSmote.prediction}
        riskScore={rfSmote.riskScore}
        groundTruth={rfSmote.groundTruth}
        outcome={rfSmote.outcome}
        threshold={rfSmote.threshold}
      />
      <ModelColumn
        label="Benchmark RF"
        showGroundTruth={showGroundTruth}
        prediction={benchmark.prediction}
        riskScore={benchmark.riskScore}
        groundTruth={benchmark.groundTruth}
        outcome={benchmark.outcome}
        threshold={benchmark.threshold}
      />
    </div>

    <ExpandedCard modelName="RF-SMOTE" {...rfSmote} />
    <ExpandedCard modelName="Benchmark RF" {...benchmark} />
  </div>
)

export default BothModelsView
