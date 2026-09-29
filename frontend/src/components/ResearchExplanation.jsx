import { useEffect, useState } from 'react'
import { api } from '../researchApi'

export default function ResearchExplanation({ explanation, model, onRetry }) {
  const item = explanation?.models?.[model]
  const [chart, setChart] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let live = true
    let url
    if (item?.waterfall_url) api(item.waterfall_url, { blob: true }).then((blob) => {
      url = URL.createObjectURL(blob)
      if (live) setChart(url)
      else URL.revokeObjectURL(url)
    }).catch((err) => { if (live) setError(err.message) })
    return () => { live = false; if (url) URL.revokeObjectURL(url) }
  }, [item?.waterfall_url])
  if (!item) return <div className="td-card" id={`waterfall_${model}`}><p role="status">{explanation?.status === 'failed'
    ? 'Explanation failed.' : 'SHAP explanation pending…'}</p>
    {explanation?.status === 'failed' && <button onClick={onRetry}>Retry explanation</button>}</div>
  return <div className="td-card" id={`waterfall_${model}`}><h3>SHAP waterfall</h3>
    <p>{item.top_positive_contributor.status === 'no_positive_contributor' ? 'No risk-increasing contributor above the numerical tolerance.'
      : `Top risk-increasing contributor: ${item.top_positive_contributor.feature}`}</p>
    {error && <p role="alert">{error}</p>}
    {chart ? <img src={chart} alt={item.chart_description} style={{ width: '100%', height: 'auto' }} /> : <p>Loading waterfall…</p>}
    <p>Reconstruction error: {item.additivity_error.toExponential(2)}</p>
  </div>
}
