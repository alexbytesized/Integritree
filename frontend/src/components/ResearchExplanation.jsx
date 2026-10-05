import { useEffect, useRef, useState } from 'react'
import { api } from '../researchApi'
import { SHAP_FEATURE_LABELS } from '../researchDisplay'
import geometry from '../../../backend/src/integritree/ml/shap_geometry.json'

const featureOrder = Object.keys(SHAP_FEATURE_LABELS)
const plotWidth = 2 * geometry.axisLimit / geometry.tickStep * geometry.tickSpacing
const svgWidth = plotWidth + 2 * geometry.endpointPadding

function ContributionChart({ item, chartApi, chartPresentation }) {
  const [chart, setChart] = useState(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const viewport = useRef(null)
  const features = [...item.features].sort((a, b) => {
    const rank = feature => featureOrder.includes(feature) ? featureOrder.indexOf(feature) : featureOrder.length
    return rank(a.feature) - rank(b.feature)
  })
  const svgHeight = geometry.topPadding + features.length * geometry.rowHeight + geometry.tickHeight
  useEffect(() => {
    let live = true
    let url
    const separator = item.waterfall_url.includes('?') ? '&' : '?'
    const query = new URLSearchParams({ layout: 'modal' })
    if (chartPresentation) query.set('presentation', chartPresentation)
    const endpoint = `${item.waterfall_url}${separator}${query}`
    chartApi(endpoint, { blob: true }).then((blob) => {
      url = URL.createObjectURL(blob)
      if (live) setChart(url)
      else URL.revokeObjectURL(url)
    }).catch((err) => { if (live) setError(err.message) })
    return () => { live = false; if (url) URL.revokeObjectURL(url) }
  }, [item.waterfall_url, attempt, chartApi, chartPresentation])
  if (error) return <><p role="alert">{error}</p><button type="button" onClick={() => { setError(''); setAttempt(v => v + 1) }}>Retry chart</button></>
  return chart ? <>
    <div className="shap-chart-section" tabIndex="0" role="region" aria-label="SHAP chart and feature names; scroll horizontally for more chart content">
      <div className="shap-chart-layout" style={{ '--shap-row-height': `${geometry.rowHeight}px`, '--shap-top-padding': `${geometry.topPadding}px` }}>
        <div className="shap-feature-labels">
          {features.map(feature => <div key={feature.feature}>{SHAP_FEATURE_LABELS[feature.feature] || feature.feature.replaceAll('_', ' ')}</div>)}
        </div>
        <div ref={viewport} className="shap-chart-scroll" tabIndex="0" role="region" aria-label="SHAP contribution plot, minus 100 to plus 100 percentage points; scroll horizontally">
          <img src={chart} style={{ width: svgWidth, height: svgHeight }} onLoad={() => {
            // Only a newly loaded chart resets the view, not ordinary React updates.
            const element = viewport.current
            if (element) element.scrollLeft = (element.scrollWidth - element.clientWidth) / 2
          }} alt={item.chart_description || 'SHAP contribution bars: green negative values decrease the fraud score; red positive values increase it. Gridlines every 10 percentage points.'} />
        </div>
        <div className="shap-direction-captions"><span>← Decrease Fraud Risk Score</span><span>Increase Fraud Risk Score →</span></div>
      </div>
    </div>
  </> : <p role="status">Loading contribution chart…</p>
}

function SummarySection({ label, value, help }) {
  return <section className="info-modal-section" aria-label={label}>
    <h3>{label}</h3>
    <p>{help}</p>
    <ul><li>{value}</li></ul>
  </section>
}

export default function ResearchExplanation({ explanation, model, onRetry, chartApi = api, chartPresentation = 'row_number' }) {
  const item = explanation?.models?.[model]
  if (!item) return <div><p role="status">{explanation?.status === 'failed'
    ? 'Explanation failed.' : 'SHAP explanation pending…'}</p>
    {explanation?.status === 'failed' && <button type="button" onClick={onRetry}>Retry explanation</button>}</div>
  const top = item.top_positive_contributor
  const topName = top?.status === 'no_positive_contributor'
    ? 'No transaction details meaningfully increased the score.'
    : SHAP_FEATURE_LABELS[top?.feature] || top?.feature || 'Unavailable'
  const percentage = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : 'Unavailable'
  return <div className="shap-explanation">
    <section className="info-modal-section" aria-label="SHAP Feature Contributions">
      <h3>SHAP Feature Contributions</h3>
      {item.waterfall_url ? <ContributionChart key={item.waterfall_url} item={item} chartApi={chartApi} chartPresentation={chartPresentation} /> : <p role="status">The contribution chart is not available yet.</p>}
    </section>
    <div className="shap-summary-sections">
      <SummarySection label="Reference" value={percentage(item.base_value)} help="The model’s average fraud score for the SHAP reference sample. This is the starting score before this transaction’s feature contributions are added." />
      <SummarySection label="Output" value={percentage(item.output_value)} help="The model’s fraud score for this transaction. It equals the reference score plus all SHAP contributions, within numerical tolerance." />
      <SummarySection label="Top Risk-Increasing Contributor" value={topName} help="The feature with the largest positive SHAP contribution above the numerical tolerance. It increased the model’s score the most; it is not a proven cause of fraud." />
    </div>
  </div>
}
