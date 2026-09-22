import "./ConfusionMatrix.css"

const MatrixCell = ({ value, label, correct }) => {
  return (
    <div className="confusion-value">
      <strong className={correct ? "confusion-value-number correct" : "confusion-value-number incorrect"}>
        {value}
      </strong>

      <span className="confusion-value-label">{label}</span>
    </div>
  )
}

const ConfusionMatrix = ({ title, truePositive, falseNegative, falsePositive, trueNegative }) => {
  return (
    <div className="confusion-matrix-wrapper">
      <div className="confusion-matrix-title-container">
        <h2>{title}</h2>
      </div>

      <div className="confusion-matrix-container">
        <div className="confusion-matrix-scroll">
          <div className="confusion-matrix">
            <div className="confusion-empty-cell" />

            <div className="confusion-table-header">Predicted Fraudulent</div>

            <div className="confusion-table-header">Predicted Legitimate</div>

            <div className="confusion-row-header">Actual Fraudulent</div>

            <MatrixCell value={truePositive} label="True Positive" correct={true} />

            <MatrixCell value={falseNegative} label="False Negative" correct={false} />

            <div className="confusion-row-header">Actual Legitimate</div>

            <MatrixCell value={falsePositive} label="False Positive" correct={false} />

            <MatrixCell value={trueNegative} label="True Negative" correct={true} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default ConfusionMatrix
