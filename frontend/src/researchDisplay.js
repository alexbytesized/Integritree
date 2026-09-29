export const MODEL_NAMES = { rf_smote: 'RF-SMOTE', rf: 'Benchmark RF' }
export const RISK_BANDS = [
  { label: 'Minimal Risk', range: 'Below 20', limit: 20, color: '#1DB954' },
  { label: 'Low Risk', range: '20 to <40', limit: 40, color: '#F9C923' },
  { label: 'Moderate Risk', range: '40 to <60', limit: 60, color: '#F47C20' },
  { label: 'High Risk', range: '60 to <80', limit: 80, color: '#D9312B' },
  { label: 'Critical Risk', range: '80 to 100', limit: Infinity, color: '#7A0000' },
]
export const riskBand = (score) => RISK_BANDS.find(band => score < band.limit) || RISK_BANDS[4]
export const scoreText = (score) => Number(score).toFixed(2)
const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
export const weekdayText = (value) => value != null && value !== '' && WEEKDAYS[value]
  ? `${value} (${WEEKDAYS[value]})` : value
export const booleanText = (value) => value === true || value === 1 || value === '1' ? '1 (True)'
  : value === false || value === 0 || value === '0' ? '0 (False)' : value

export function featureText(feature, derived = {}) {
  const value = derived[feature.feature]
  if (feature.feature === 'day_of_week' && value != null) return `Simulated weekday: ${weekdayText(value)}`
  if (feature.feature === 'hour_of_day' && value != null) return `Simulated hour: ${value}`
  return (feature.readable_value || feature.feature.replaceAll('_', ' '))
    .replaceAll('CASH_IN', 'cash in').replaceAll('CASH_OUT', 'cash out')
    .replaceAll('TRANSFER', 'transfer').replaceAll('PAYMENT', 'payment').replaceAll('DEBIT', 'debit')
}

export function shapSummary(explanation, model, derived) {
  const item = explanation?.models?.[model]
  if (!item) return explanation?.status === 'failed'
    ? 'The explanation could not be loaded. Open the full SHAP evaluation to retry.'
    : 'The SHAP explanation is being prepared.'
  if (!item.features?.length) return 'No feature contributions are available for this record.'
  const describe = (direction) => {
    const factors = item.features.filter(f => direction * f.contribution > 1e-9)
      .sort((a, b) => direction * (b.contribution - a.contribution)).slice(0, 2)
    const verb = direction === 1 ? 'raised' : 'lowered'
    if (!factors.length) return `No transaction details meaningfully ${verb} the score.`
    return `The main details that ${verb} the score were: ${factors.map(f =>
      `${featureText(f, derived)} (${direction === 1 ? '+' : '−'}${scoreText(Math.abs(f.contribution) * 100)} percentage points)`
    ).join('; ')}.`
  }
  return `SHAP shows how transaction details raised or lowered this model’s fraud score. ${describe(1)} ${describe(-1)} These contributions explain the model’s score; they do not prove that a detail caused fraud.`
}
