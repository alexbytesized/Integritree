import { useEffect, useState } from 'react'
import { api } from '../researchApi'
import { featureText } from '../researchDisplay'

function WaterfallChart({ item }) {
  const [chart, setChart] = useState(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let live = true
    let url
    const separator = item.waterfall_url.includes('?') ? '&' : '?'
    api(`${item.waterfall_url}${separator}presentation=row_number`, { blob: true }).then((blob) => {
      url = URL.createObjectURL(blob)
      if (live) setChart(url)
      else URL.revokeObjectURL(url)
    }).catch((err) => { if (live) setError(err.message) })
    return () => { live = false; if (url) URL.revokeObjectURL(url) }
  }, [item.waterfall_url, attempt])
  if (error) return <><p role="alert">{error}</p><button type="button" onClick={() => { setError(''); setAttempt(v => v + 1) }}>Retry chart</button></>
  return chart ? <div className="shap-chart-scroll" tabIndex="0" role="region" aria-label="SHAP waterfall chart; scroll horizontally for details">
    <img src={chart} alt={item.chart_description || 'SHAP waterfall showing how transaction details changed the model score.'} />
  </div> : <p role="status">Loading waterfall…</p>
}

export default function ResearchExplanation({ explanation, model, onRetry }) {
  const item = explanation?.models?.[model]
  if (!item) return <div><p role="status">{explanation?.status === 'failed'
    ? 'Explanation failed.' : 'SHAP explanation pending…'}</p>
    {explanation?.status === 'failed' && <button type="button" onClick={onRetry}>Retry explanation</button>}</div>
  const top = item.features?.find(f => f.feature === item.top_positive_contributor?.feature)
  return <div><h3>SHAP waterfall</h3>
    <p>{item.top_positive_contributor?.status === 'no_positive_contributor' ? 'No transaction details meaningfully increased the score.'
      : `Top risk-increasing contributor: ${top ? featureText(top) : item.top_positive_contributor?.feature || 'Unavailable'}`}</p>
    {item.waterfall_url ? <WaterfallChart key={item.waterfall_url} item={item} /> : <p role="status">The waterfall chart is not available yet.</p>}
    {item.additivity_error != null && <p>Reconstruction error: {item.additivity_error.toExponential(2)}</p>}
  </div>
}
