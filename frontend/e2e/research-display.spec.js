import { test, expect } from '@playwright/test'
import { booleanText, weekdayText, riskBand, scoreText, shapSummary } from '../src/researchDisplay.js'

const fingerprint = '23c03e6b' + 'a'.repeat(56)
const makeItem = model => ({
  waterfall_url: `/api/v1/research/analyses/display/records/1/waterfall/${model}`,
  chart_description: `${model} waterfall from reference to prediction`, additivity_error: 0,
  features: [
    { feature: 'hour_of_day', readable_value: 'Simulation hour = 5', contribution: .3995 },
    { feature: 'day_of_week', readable_value: 'Simulation day index = 0', contribution: -.05118 },
  ],
  top_positive_contributor: { status: 'available', feature: 'hour_of_day' },
})
const recordFixture = () => ({
  row_number: 1, transaction_id: `${fingerprint}:1`, actual_label: 1, threshold: .43,
  rf_smote: { score: .535176, predicted_label: 1 }, rf: { score: .145, predicted_label: 0 },
  original: { step: 6, type: 'CASH_OUT', amount: 100, nameOrig: 'C_DEMO', nameDest: 'M_DEMO' },
  derived: { hour_of_day: 5, day_of_week: 0, type_CASH_IN: 0, type_CASH_OUT: 1, type_DEBIT: 0,
    type_PAYMENT: 0, type_TRANSFER: 0, log_amount: 4.615, is_zero_amount: 0, is_merchant_origin: 0, is_merchant_dest: 1 },
  explanation: { status: 'computed', models: { rf: makeItem('rf'), rf_smote: makeItem('rf_smote') } },
})

async function setup(page, record, options = {}) {
  const chartRequests = []
  await page.route('**/api/v1/research/analyses/display/**', async route => {
    const url = new URL(route.request().url())
    if (url.pathname.includes('/waterfall/')) {
      chartRequests.push(url)
      if (options.chartFailure) {
        options.chartFailure = false
        return route.fulfill({ status: 503, json: { detail: 'Chart temporarily unavailable.' } })
      }
      return route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700"><rect width="1000" height="700" fill="white"/><text x="40" y="60">Transaction 1 SHAP waterfall</text></svg>' })
    }
    if (route.request().method() === 'POST') {
      record.explanation = { status: 'computed', models: { rf: makeItem('rf'), rf_smote: makeItem('rf_smote') } }
      return route.fulfill({ json: { status: 'pending' } })
    }
    return route.fulfill({ json: record })
  })
  await page.goto('/researcher/transaction/1?analysis=display')
  await expect(page.locator('.td-record-id')).toHaveText('Transaction ID: 1')
  return chartRequests
}

test('input formatting and exact risk-band boundaries', () => {
  expect(weekdayText(0)).toBe('0 (Monday)')
  expect(weekdayText(6)).toBe('6 (Sunday)')
  expect(weekdayText(null)).toBeNull()
  for (const value of [0, false, '0']) expect(booleanText(value)).toBe('0 (False)')
  for (const value of [1, true, '1']) expect(booleanText(value)).toBe('1 (True)')
  expect(booleanText(null)).toBeNull()
  expect(booleanText(5)).toBe(5)
  expect([0, 53.5176, 100].map(scoreText)).toEqual(['0.00', '53.52', '100.00'])
  const cases = [[0, 'Minimal'], [19.9999, 'Minimal'], [20, 'Low'], [39.9999, 'Low'],
    [40, 'Moderate'], [59.9999, 'Moderate'], [60, 'High'], [79.9999, 'High'], [80, 'Critical'], [100, 'Critical']]
  for (const [score, band] of cases) expect(riskBand(score).label).toBe(`${band} Risk`)
  expect(shapSummary({ models: { rf: { features: [{ contribution: 0 }] } } }, 'rf'))
    .toContain('No transaction details meaningfully raised the score. No transaction details meaningfully lowered the score.')
})

test('all model views use display IDs, colored interpretations and model-specific modals', async ({ page }, testInfo) => {
  const requests = await setup(page, recordFixture())
  await expect(page.locator('body')).not.toContainText(fingerprint)
  const field = label => page.locator('.td-field-row').filter({ hasText: label }).locator('.td-field-value')
  await expect(field('Day of the Week')).toHaveText('0 (Monday)')
  await expect(field('Transaction is Fraudulent:')).toHaveText('1 (True)')
  await expect(field('Transaction is Cash In:')).toHaveText('0 (False)')
  await expect(field('Transaction is Cash Out:')).toHaveText('1 (True)')
  await expect(field('Origin Old Balance:')).toHaveText('Not supplied')
  for (const [name, score, prediction, band, model] of [
    ['RF-SMOTE', '53.52', 'Fraudulent', 'Moderate Risk', 'rf_smote'],
    ['Benchmark RF', '14.50', 'Legitimate', 'Minimal Risk', 'rf'],
  ]) {
    await page.getByRole('tab', { name, exact: true }).click()
    await expect(page.locator('.mrv-risk-score')).toHaveText(`${score} out of 100`)
    const interpretation = page.locator('.mrv-text').first()
    await expect(interpretation).toHaveText(`The ${name} model classified the transaction as ${prediction}, with a fraud risk score of ${score}%, corresponding to a ${band} level.`)
    await expect(interpretation.locator('strong').last()).toHaveCSS('color', model === 'rf_smote' ? 'rgb(244, 124, 32)' : 'rgb(29, 185, 84)')
    await expect(page.locator('.mrv-text').last()).toContainText('39.95 percentage points')
    await expect(page.locator('.mrv-text').last()).toContainText('5.12 percentage points')
    await expect(page.getByRole('dialog')).toHaveCount(0)
    const before = requests.length
    const open = page.getByRole('button', { name: 'See Full SHAP Evaluation' })
    await open.click()
    await expect(page.getByRole('dialog', { name: `${name} — SHAP Evaluation` })).toBeVisible()
    await expect(page.getByRole('dialog').locator('img')).toBeVisible()
    expect(requests.length).toBe(before + 1)
    expect(requests.at(-1).pathname).toContain(`/waterfall/${model}`)
    expect(requests.at(-1).searchParams.get('presentation')).toBe('row_number')
    await expect(page.getByRole('button', { name: 'Close SHAP evaluation' })).toBeFocused()
    await page.keyboard.press('Shift+Tab')
    await expect(page.getByRole('region', { name: 'SHAP waterfall chart; scroll horizontally for details', exact: true })).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(open).toBeFocused()
    await expect(page.getByRole('dialog')).toHaveCount(0)
  }
  await page.getByRole('tab', { name: 'Both Models' }).click()
  await expect(page.getByRole('button', { name: 'View risk classification thresholds' })).toHaveCount(2)
  for (const [index, name] of ['RF-SMOTE', 'Benchmark RF'].entries()) {
    await page.getByRole('button', { name: 'See Full SHAP Evaluation' }).nth(index).click()
    await expect(page.getByRole('dialog', { name: `${name} — SHAP Evaluation` })).toBeVisible()
    await page.locator('.info-modal-overlay').click({ position: { x: 2, y: 2 } })
    await expect(page.getByRole('dialog')).toHaveCount(0)
  }
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: testInfo.outputPath('both-models.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: 'See Full SHAP Evaluation' }).first().click()
  await expect(page.getByRole('dialog').locator('img')).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('mobile-modal.png') })
  const box = await page.getByRole('dialog').boundingBox()
  expect(box.x).toBeGreaterThanOrEqual(0)
  expect(box.x + box.width).toBeLessThanOrEqual(390)
})

test('overlapping markers, tooltips and risk-level note remain accessible', async ({ page }, testInfo) => {
  const record = recordFixture()
  record.rf_smote.score = .43
  await setup(page, record)
  await page.getByRole('tab', { name: 'RF-SMOTE', exact: true }).click()
  const score = page.getByRole('button', { name: 'Risk Score: 43.00%', exact: true })
  const threshold = page.getByRole('button', { name: 'Fraud Threshold Line: 43.00%', exact: true })
  await score.hover()
  await expect(page.getByRole('tooltip')).toHaveText('Risk Score: 43.00%')
  const lineBounds = await threshold.boundingBox()
  await page.mouse.click(lineBounds.x + lineBounds.width / 2, lineBounds.y + lineBounds.height - 3)
  await expect(page.getByRole('tooltip')).toHaveText('Fraud Threshold Line: 43.00%')
  await threshold.press('Escape')
  await expect(page.getByRole('tooltip')).toHaveCount(0)
  await score.focus()
  await expect(page.getByRole('tooltip')).toHaveText('Risk Score: 43.00%')
  await page.getByRole('button', { name: 'View risk classification thresholds' }).focus()
  await page.keyboard.press('Enter')
  await expect(page.locator('.risk-tooltip-note')).toHaveText('Note: The following bands describe model scores, not calibrated real-world probabilities.')
  await expect(page.locator('.risk-tooltip-note')).toHaveCSS('font-size', '12px')
  await expect(page.locator('.risk-tooltip-note')).toHaveCSS('margin-top', '16px')
  await page.screenshot({ path: testInfo.outputPath('risk-tooltip.png'), fullPage: true })
})

test('pending, explanation retry and chart retry work inside the modal', async ({ page }) => {
  const record = recordFixture()
  record.explanation = { status: 'pending' }
  await setup(page, record, { chartFailure: true })
  await page.getByRole('tab', { name: 'RF-SMOTE', exact: true }).click()
  await page.getByRole('button', { name: 'See Full SHAP Evaluation' }).click()
  await expect(page.getByRole('dialog')).toContainText('SHAP explanation pending')
  record.explanation = { status: 'failed' }
  await page.getByRole('button', { name: 'Retry explanation' }).click()
  await expect(page.getByRole('dialog').getByRole('alert')).toHaveText('Chart temporarily unavailable.')
  await page.getByRole('button', { name: 'Retry chart' }).click()
  await expect(page.getByRole('dialog').locator('img')).toBeVisible()
  await page.getByRole('button', { name: 'Close SHAP evaluation' }).click()
  await expect(page.getByRole('dialog')).toHaveCount(0)
})

test('table badges fit boundary scores without changing gaps, and search requests row numbers', async ({ page }, testInfo) => {
  const metrics = Object.fromEntries(['precision', 'recall', 'f1', 'mcc', 'pr_auc'].map(key => [key, { value: 1 }]))
  const model = { metrics, actual_fraud: 1, confusion_matrix: { tp: 1, tn: 2, fp: 0, fn: 0 } }
  const job = { status: 'complete', filename: 'display-fixture.csv', rows_processed: 3, threshold: .43,
    evaluation: { models: { rf: model, rf_smote: model }, descriptive_comparisons: metrics,
      statistical_test: { p_value: 1, table: { both_correct: 3, rf_only_correct: 0, smote_only_correct: 0, both_incorrect: 0 } } } }
  const records = [0, .5352, 1].map((score, i) => ({ ...recordFixture(), row_number: i + 1,
    transaction_id: `${fingerprint}:${i + 1}`, rf: { score, predicted_label: 0 }, rf_smote: { score, predicted_label: 1 } }))
  const searches = []
  await page.route('**/api/v1/research/analyses/table**', async route => {
    const url = new URL(route.request().url())
    if (url.pathname.endsWith('/records')) {
      searches.push(url)
      const filtered = records.filter(row => String(row.row_number).includes(url.searchParams.get('search') || ''))
      return route.fulfill({ json: { records: filtered, total: filtered.length, page: 1 } })
    }
    return route.fulfill({ json: job })
  })
  await page.goto('/researcher?analysis=table')
  await expect(page.locator('.transactionTable tbody th')).toHaveText(['1', '2', '3'])
  await expect(page.locator('.riskBadge.blueBadge')).toHaveText(['0.00%', '53.52%', '100.00%'])
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 })
    const badge = page.locator('.riskBadge').last()
    await badge.scrollIntoViewIfNeeded()
    await expect(badge).toHaveCSS('min-width', '92px')
    await expect(badge.locator('..')).toHaveCSS('gap', '14px')
    const fit = await badge.evaluate(element => {
      const range = document.createRange()
      range.selectNodeContents(element)
      const text = range.getBoundingClientRect(), box = element.getBoundingClientRect()
      return text.left > box.left && text.right < box.right
    })
    expect(fit).toBe(true)
    await page.locator('.tableContainer').screenshot({ path: testInfo.outputPath(`table-${width}.png`) })
  }
  await page.getByPlaceholder('Search Transaction ID').fill('3')
  await expect(page.locator('.transactionTable tbody th')).toHaveText(['3'])
  expect(searches.at(-1).searchParams.get('search_field')).toBe('row_number')
  await expect(page.getByRole('button', { name: 'View transaction 3', exact: true })).toBeVisible()
})
