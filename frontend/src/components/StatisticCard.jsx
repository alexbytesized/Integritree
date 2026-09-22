import "./StatisticCard.css"
import InfoButton from "./InfoButton"

const StatisticCard = ({ title, precision, recall, f1, mcc, auc, onRequestInfo }) => {
  const metrics = [
    ["Precision", precision],
    ["Recall", recall],
    ["F1-Score", f1],
    ["MCC", mcc],
    ["PR-AUC", auc],
  ]

  return (
    <div className="statistic-result-container">
      <div className="statistic-result-title">
        <h3>{title}</h3>
      </div>

      <div className="statistic-result-card">
        {metrics.map(([label, value]) => (
          <div className="statistic-row" key={label}>
            <div className="statistic-label">
              <InfoButton topic={label} onRequestInfo={onRequestInfo} />
              <strong>{label}:</strong>
            </div>
            <span>{value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default StatisticCard
