import "./PercentageDifference.css"

const PercentageDifferenceCard = ({ percentage, label }) => {
  const negative = percentage.startsWith("-")
  return (
    <div className={`difference-card${negative ? " is-negative" : ""}`}>
      <span>
        <strong>{percentage}</strong> difference
      </span>
      <strong className="difference-label">{label}</strong>
    </div>
  )
}

export default PercentageDifferenceCard
