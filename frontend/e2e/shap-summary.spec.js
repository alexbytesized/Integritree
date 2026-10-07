import { test, expect } from '@playwright/test'
import { shapSummary, shapSummaryParts } from '../src/researchDisplay.js'

const explanation = (values, overrides = {}) => ({ status: 'computed', models: { rf: {
  output_value: .82, predicted_label: 1, base_value: .1, threshold: .43,
  features: Object.entries(values).map(([feature, contribution]) => ({ feature, contribution })),
  ...overrides,
} } })
const factors = (value, context) => shapSummaryParts(value, 'rf', context).filter(part => part.bold && !part.color).slice(1).map(part => part.text)

test('signed grouping changes the leading factor and omits net lowering groups', () => {
  const value = explanation({ type_TRANSFER: .4, type_PAYMENT: -.35, log_amount: .2, is_zero_amount: -.08, hour_of_day: -.2 })
  expect(factors(value)).toEqual(['The transaction amount'])
  expect(shapSummary(value, 'rf')).toBe('The model classified this transaction as Fraudulent, with a fraud risk score of 82.00/100, which falls under the category of Critical Risk. The transaction amount increased the score the most.')
})

test('always includes the strongest positive group even below ten points', () => {
  expect(factors(explanation({ log_amount: .00001, hour_of_day: .000001 }))).toEqual(['The transaction amount'])
  expect(factors(explanation({ log_amount: 1e-9 }))).toEqual([])
  expect(factors(explanation({ log_amount: 1.01e-9 }))).toEqual(['The transaction amount'])
})

test('additional groups must exceed ten points before rounding; include all qualifying groups', () => {
  const value = explanation({ type_TRANSFER: .3, log_amount: .2, day_of_week: .10000001,
    hour_of_day: .1, is_merchant_origin: .09999999, is_merchant_dest: .11 })
  expect(factors(value)).toEqual(['The transaction type', 'The transaction amount', 'the recipient type', 'the simulated day of the week'])
  expect(shapSummary(value, 'rf')).toContain('The transaction amount, the recipient type, and the simulated day of the week also increased the score.')
})

test('ties follow group order regardless of incoming feature order', () => {
  expect(factors(explanation({ is_merchant_dest: .2, is_merchant_origin: .2, hour_of_day: .2,
    day_of_week: .2, log_amount: .2, type_TRANSFER: .2 }))).toEqual([
    'The transaction type', 'The transaction amount', 'the simulated day of the week',
    'the simulated hour of the day', 'the sender type', 'the recipient type',
  ])
})

test('legitimate wording and Manila context use the supplied model result', () => {
  const value = explanation({ hour_of_day: .2 })
  const text = shapSummary(value, 'rf', 'receipt', { score: .28, predicted_label: 0 })
  expect(text).toContain('Legitimate, with a fraud risk score of 28.00/100, which falls under the category of Low Risk.')
  expect(text).toContain('Although the hour of the day in Manila time increased the score, the final score remained below the model’s fraud classification cutoff.')
  expect(factors(explanation({ day_of_week: .2 }), 'receipt')).toEqual(['The day of the week in Manila time'])
})

test('no positive groups handles either prediction, including a high reference score', () => {
  for (const [score, prediction] of [[.2, 0], [.82, 1]]) {
    const value = explanation({ log_amount: -.02, type_TRANSFER: .01, type_PAYMENT: -.01 },
      { base_value: score + .02, output_value: score, predicted_label: prediction })
    const text = shapSummary(value, 'rf')
    expect(text).toContain('None of the transaction factors increased the score above the model’s reference score.')
    expect(text.includes('The reference score was already above the classification cutoff.')).toBe(prediction === 1)
    expect(factors(value)).toEqual([])
  }
})

test('missing, pending, failed and invalid inputs do not manufacture an explanation', () => {
  expect(shapSummary(undefined, 'rf')).toBe('The SHAP explanation is being prepared.')
  expect(shapSummary({ status: 'pending' }, 'rf')).toBe('The SHAP explanation is being prepared.')
  expect(shapSummary({ status: 'failed' }, 'rf')).toContain('Open the full SHAP evaluation to retry.')
  expect(shapSummary(explanation({}), 'rf')).toBe('No feature contributions are available for this record.')
  expect(shapSummary(explanation({ log_amount: NaN }), 'rf')).toBe('No feature contributions are available for this record.')
  expect(shapSummary(explanation({ unknown: .5 }), 'rf')).toBe('No feature contributions are available for this record.')
  expect(shapSummary(explanation({ log_amount: .1 }, { output_value: undefined }), 'rf')).toContain('prediction and score are not available')
})
