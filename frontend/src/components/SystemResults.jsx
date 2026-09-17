import "./SystemResults.css"
import FraudDonut from "../components/FraudDonut"

const SystemResults = ({ title, legitimateRecords, fraudulentRecords, fraudRate, titleClass }) => {
  return (
    <div className="model-result-card">
      <h3 className={titleClass}>{title}</h3>

      <div className="model-stat">
        <strong className="legitimate-value">{legitimateRecords}</strong>
        <span>Legitimate Records</span>
      </div>

      <div className="model-stat">
        <strong className="fraudulent-value">{fraudulentRecords}</strong>
        <span>Fraudulent Records</span>
      </div>

      <div className="fraud-rate">
        <div className="fraud-chart">
          <FraudDonut value={fraudRate} />
        </div>
      </div>
    </div>
  )
}

export default SystemResults
