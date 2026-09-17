import "./PercentageDifference.css"

const PercentageDifferenceCard = ({ percentage, label }) => {
  return (
    <div className="difference-card">
      <span>
        <strong>{percentage}</strong> difference
      </span>
      <h2>{label}</h2>
    </div>
  )
}

export default PercentageDifferenceCard
