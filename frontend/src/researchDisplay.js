export const MODEL_NAMES = { rf_smote: 'RF-SMOTE', rf: 'Benchmark RF' }
export const SHAP_FEATURE_LABELS = {
  hour_of_day: 'Hour of the Day', day_of_week: 'Day of the Week',
  type_CASH_IN: 'Transaction is Cash In', type_CASH_OUT: 'Transaction is Cash Out',
  type_DEBIT: 'Transaction is Debit', type_PAYMENT: 'Transaction is Payment',
  type_TRANSFER: 'Transaction is Transfer', log_amount: 'Transaction Log Amount',
  is_zero_amount: 'Transaction Amount is 0', is_merchant_origin: 'Origin is Merchant',
  is_merchant_dest: 'Destination is Merchant',
}
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

export function featureText(feature, derived = {}, context = 'research') {
  const value = derived[feature.feature]
  if (feature.feature === 'day_of_week' && value != null) return `${context === 'receipt' ? 'Manila' : 'Simulated'} weekday: ${weekdayText(value)}`
  if (feature.feature === 'hour_of_day' && value != null) return `${context === 'receipt' ? 'Manila' : 'Simulated'} hour: ${value}`
  return (feature.readable_value || feature.feature.replaceAll('_', ' '))
    .replaceAll('CASH_IN', 'cash in').replaceAll('CASH_OUT', 'cash out')
    .replaceAll('TRANSFER', 'transfer').replaceAll('PAYMENT', 'payment').replaceAll('DEBIT', 'debit')
}

// Group signed contributions before ranking: individual indicators can offset
// one another. This order also provides a stable tie-breaker.
const SUMMARY_GROUPS = [
  ['transaction type', ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']],
  ['transaction amount', ['log_amount', 'is_zero_amount']],
  ['day', ['day_of_week']],
  ['hour', ['hour_of_day']],
  ['sender type', ['is_merchant_origin']],
  ['recipient type', ['is_merchant_dest']],
]

export function shapSummaryParts(explanation, model, context = 'research', result) {
  const item = explanation?.models?.[model]
  const message = text => [{ text }]
  if (!item) return message(explanation?.status === 'failed'
    ? 'The explanation could not be loaded. Open the full SHAP evaluation to retry.'
    : 'The SHAP explanation is being prepared.')
  if (!item.features?.length) return message('No feature contributions are available for this record.')
  const score = result?.score ?? item.output_value
  const prediction = result?.predicted_label ?? item.predicted_label
  if (!Number.isFinite(score) || score < 0 || score > 1 || ![0, 1].includes(prediction)) {
    return message('The prediction and score are not available for this explanation.')
  }
  if (item.features.some(feature => !Number.isFinite(feature.contribution)
    || !SUMMARY_GROUPS.some(([, names]) => names.includes(feature.feature)))) {
    return message('No feature contributions are available for this record.')
  }
  const groups = SUMMARY_GROUPS.map(([name, names], order) => ({
    name: name === 'day' ? (context === 'receipt' ? 'day of the week in Manila time' : 'simulated day of the week')
      : name === 'hour' ? (context === 'receipt' ? 'hour of the day in Manila time' : 'simulated hour of the day') : name,
    contribution: item.features.filter(feature => names.includes(feature.feature))
      .reduce((sum, feature) => sum + feature.contribution, 0),
    order,
  })).filter(group => group.contribution > 1e-9)
    .sort((a, b) => b.contribution - a.contribution || a.order - b.order)
  const selected = groups.filter((group, index) => index === 0 || group.contribution > 0.10)
  const band = riskBand(score * 100)
  const parts = [
    { text: 'The model classified this transaction as ' },
    { text: prediction === 1 ? 'Fraudulent' : 'Legitimate', bold: true, color: prediction === 1 ? '#C0392B' : '#1DB954' },
    { text: ', with a fraud risk score of ' },
    { text: `${scoreText(score * 100)}/100`, bold: true },
    { text: ', which falls under the category of ' },
    { text: band.label, bold: true, color: band.color },
    { text: '. ' },
  ]
  if (!selected.length) {
    parts.push({ text: 'None of the transaction factors increased the score above the model’s reference score.' })
    if (prediction === 1) parts.push({ text: ' The reference score was already above the classification cutoff.' })
    return parts
  }
  if (prediction === 0) parts.push({ text: 'Although ' })
  parts.push({ text: `${prediction === 1 ? 'The' : 'the'} ${selected[0].name}`, bold: true })
  parts.push({ text: prediction === 1 ? ' increased the score the most.'
    : ' increased the score, the final score remained below the model’s fraud classification cutoff.' })
  selected.slice(1).forEach((group, index, others) => {
    parts.push({ text: index === 0 ? ' ' : index === others.length - 1 ? (others.length > 2 ? ', and ' : ' and ') : ', ' })
    parts.push({ text: `${index === 0 ? 'The' : 'the'} ${group.name}`, bold: true })
    if (index === others.length - 1) parts.push({ text: ' also increased the score.' })
  })
  return parts
}

export function shapSummary(explanation, model, context = 'research', result) {
  return shapSummaryParts(explanation, model, context, result).map(part => part.text).join('')
}
