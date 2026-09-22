import { Info } from "lucide-react"

const InfoButton = ({ topic, onRequestInfo }) => (
  <button
    type="button"
    className="info-button"
    aria-label={`About ${topic}`}
    title="Explanation coming soon"
    onClick={() => onRequestInfo?.(topic)}
  >
    <Info aria-hidden="true" />
  </button>
)

export default InfoButton
