import { Info } from "lucide-react"

const InfoButton = ({ topic, onRequestInfo, label = `About ${topic}` }) => (
  <button
    type="button"
    className="info-button"
    aria-label={label}
    title={label}
    onClick={() => onRequestInfo?.(topic)}
  >
    <Info aria-hidden="true" />
  </button>
)

export default InfoButton
