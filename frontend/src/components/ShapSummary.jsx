import { shapSummaryParts } from '../researchDisplay'

export default function ShapSummary({ explanation, model, context = 'research', result }) {
  return <span className="shap-summary">{shapSummaryParts(explanation, model, context, result).map((part, index) =>
    part.bold ? <strong key={index} style={part.color ? { color: part.color } : undefined}>{part.text}</strong>
      : part.text
  )}</span>
}
