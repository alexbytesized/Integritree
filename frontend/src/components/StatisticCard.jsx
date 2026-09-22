import "./StatisticCard.css"

const StatisticCard = ({ title, precision, recall, f1, mcc, auc, titleClass }) => {
  return (
    <div className="statistic-result-container">
      <div className="statistic-result-title">
        <h1>{title}</h1>
      </div>

      <div className="statistic-result-card">
        <div className="statistic-row">
          <h2>Precision:</h2>
          <span>{precision}</span>
        </div>

        <div className="statistic-row">
          <h2>Recall:</h2>
          <span>{recall}</span>
        </div>

        <div className="statistic-row">
          <h2>F1-Score:</h2>
          <span>{f1}</span>
        </div>

        <div className="statistic-row">
          <h2>MCC:</h2>
          <span>{mcc}</span>
        </div>

        <div className="statistic-row">
          <h2>PR-AUC:</h2>
          <span>{auc}</span>
        </div>
      </div>
    </div>
  )
}

export default StatisticCard
