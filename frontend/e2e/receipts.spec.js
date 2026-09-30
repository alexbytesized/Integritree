import { test, expect } from '@playwright/test'

const image = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64')
const fields = { workflow: 'express_send', category: 'TRANSFER', amount: '100.00', date: '2026-10-05', time: '13:00',
  reference: null, origin_role: 'personal_wallet', destination_role: 'personal_wallet', wallet_funded: true }
const derived = { hour_of_day: 13, day_of_week: 0, type_TRANSFER: 1, type_PAYMENT: 0, type_DEBIT: 0,
  type_CASH_IN: 0, type_CASH_OUT: 0, log_amount: 4.615, is_zero_amount: 0, is_merchant_origin: 0, is_merchant_dest: 0 }
const makeExplanation = revision => ({ status: 'computed', models: Object.fromEntries(['rf', 'rf_smote'].map(model => [model, {
  waterfall_url: `/api/v1/receipts/demo/waterfall/${model}?revision=${revision}`, chart_description: `${model} receipt waterfall`,
  additivity_error: 0, features: [{ feature: 'hour_of_day', readable_value: 'Manila hour = 13', contribution: .1 }],
  top_positive_contributor: { status: 'available', feature: 'hour_of_day' },
}])) })

async function mockReceipts(page, unsupported = false) {
  const job = { id: 'demo', filename: 'receipt.png', status: unsupported ? 'unsupported' : 'awaiting_confirmation',
    busy: false, revision: 0, workflow: unsupported ? undefined : 'express_send', candidates: { ...fields, date: null },
    error: unsupported ? 'The receipt layout is unclear or unsupported.' : null }
  const confirmations = [], charts = []
  await page.route('**/api/v1/receipts**', async route => {
    const request = route.request(), url = new URL(request.url())
    const path = url.pathname
    if (path.endsWith('/sessions')) return route.fulfill({ json: { token: 'receipt-test' } })
    if (request.method() === 'DELETE') return route.fulfill({ status: 204 })
    if (path.endsWith('/image')) return route.fulfill({ contentType: 'image/png', body: image })
    if (path.includes('/waterfall/')) {
      charts.push({ url, token: request.headers()['x-receipt-session'] })
      return route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700"><text x="20" y="30">Receipt SHAP</text></svg>' })
    }
    if (path.endsWith('/download')) return route.fulfill({ contentType: 'application/zip', body: Buffer.from('synthetic zip') })
    if (path.endsWith('/confirm')) {
      const input = request.postDataJSON()
      confirmations.push(input)
      job.revision++
      job.status = 'complete'
      job.confirmed_fields = input.fields
      job.result = { original: input.fields, derived, revision: job.revision, threshold: .43,
        rf: { score: .125, predicted_label: 0 }, rf_smote: { score: .535176, predicted_label: 1 },
        explanation: makeExplanation(job.revision) }
      return route.fulfill({ status: 202, json: { id: 'demo', status: 'predicting', revision: job.revision } })
    }
    if (request.method() === 'POST') return route.fulfill({ status: 202, json: { id: 'demo', status: 'queued' } })
    return route.fulfill({ json: job })
  })
  return { confirmations, charts }
}

async function begin(page, sample = null) {
  await page.goto('/upload')
  await page.locator('input[type=file]').setInputFiles(sample || { name: 'receipt.png', mimeType: 'image/png', buffer: image })
  await page.getByRole('button', { name: 'Extract transaction details' }).click()
  await expect(page.getByRole('heading', { name: 'Review your receipt' })).toBeVisible()
}

async function confirm(page) {
  await page.getByLabel('Transaction date (Manila)').fill('2026-10-05')
  await page.getByLabel(/I confirm this completed transaction/).check()
  await page.getByLabel('I have reviewed and confirm these transaction details.').check()
  await page.getByRole('button', { name: 'Confirm and analyze' }).click()
  await expect(page.getByRole('heading', { name: 'Results Overview' })).toBeVisible()
}

test('receipt confirmation, refresh, user model designs without labels, SHAP, revision and Clear', async ({ page }, testInfo) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const state = await mockReceipts(page)
  await begin(page)
  await expect(page.getByRole('button', { name: 'Confirm and analyze' })).toBeDisabled()
  await expect(page.getByLabel('Transaction date (Manila)')).toHaveValue('')
  await confirm(page)
  expect(state.confirmations[0].expected_revision).toBe(0)
  expect(state.confirmations[0].fields.reference).toBeNull()
  await page.reload()
  await expect(page.getByText('Original Inputs', { exact: true })).toBeVisible()
  await expect(page.getByText('Derived Inputs', { exact: true })).toBeVisible()
  await expect(page.getByText('0 (Monday)', { exact: true })).toBeVisible()
  for (const tab of ['RF-SMOTE', 'Benchmark RF', 'Both Models']) {
    await page.getByRole('tab', { name: tab, exact: true }).click()
    await expect(page.getByText('Ground Truth', { exact: true })).toHaveCount(0)
    await expect(page.getByText('Outcome', { exact: true })).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Fraud Threshold Line: 43.00%' })).toHaveCount(tab === 'Both Models' ? 2 : 1)
    await page.getByRole('button', { name: 'See Full SHAP Evaluation' }).first().click()
    await expect(page.getByRole('dialog').locator('img')).toBeVisible()
    expect(state.charts.at(-1).token).toBe('receipt-test')
    expect(state.charts.at(-1).url.searchParams.get('revision')).toBe('1')
    expect(state.charts.at(-1).url.searchParams.has('presentation')).toBe(false)
    await page.keyboard.press('Escape')
  }
  await page.screenshot({ path: testInfo.outputPath('receipt-models-desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('receipt-models-mobile.png'), fullPage: true })
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download results' }).click()
  expect((await download).suggestedFilename()).toBe('integritree_receipt.zip')
  await page.getByRole('link', { name: 'Edit confirmed details' }).click()
  await page.getByLabel('Principal amount (PHP)').fill('200.00')
  await confirm(page)
  expect(state.confirmations[1].expected_revision).toBe(1)
  expect(state.confirmations[1].fields.amount).toBe('200.00')
  await page.getByRole('button', { name: 'Clear results' }).click()
  await expect(page).toHaveURL(/\/upload$/)
  expect(await page.evaluate(() => sessionStorage.getItem('integritree-receipt-id'))).toBeNull()
  expect(errors).toEqual([])
})

test('unsupported receipt cannot expose a confirmation form or fabricated results', async ({ page }) => {
  await mockReceipts(page, true)
  await begin(page)
  await expect(page.getByRole('alert')).toContainText('unsupported')
  await expect(page.getByRole('button', { name: 'Confirm and analyze' })).toHaveCount(0)
  await page.goto('/user-results?receipt=demo')
  await expect(page.getByText('No confirmed receipt results yet.')).toHaveCount(0)
  await expect(page.getByRole('alert')).toContainText('unsupported')
  await expect(page.getByText('75 out of 100')).toHaveCount(0)
})

test('live local OCR, saved prediction, SHAP, download and cleanup', async ({ page }, testInfo) => {
  test.skip(!process.env.RECEIPT_SAMPLE_PATH, 'Opt-in local sample; never commit private screenshots.')
  test.setTimeout(240000)
  if (process.env.RECEIPT_API_BASE_URL) await page.route('**/api/v1/receipts**', async route => {
    const url = new URL(route.request().url())
    const response = await route.fetch({ url: `${process.env.RECEIPT_API_BASE_URL}${url.pathname}${url.search}` })
    await route.fulfill({ response })
  })
  await begin(page, process.env.RECEIPT_SAMPLE_PATH)
  await expect(page.getByLabel('Principal amount (PHP)')).not.toHaveValue('')
  // Keep actual extracted values; only complete a missing time/date for this UI check.
  if (!await page.getByLabel('Transaction date (Manila)').inputValue()) await page.getByLabel('Transaction date (Manila)').fill('2026-10-05')
  if (!await page.getByLabel('Transaction time (Manila)').inputValue()) await page.getByLabel('Transaction time (Manila)').fill('13:00')
  await page.getByLabel(/I confirm this completed transaction/).check()
  await page.getByLabel('I have reviewed and confirm these transaction details.').check()
  await page.getByRole('button', { name: 'Confirm and analyze' }).click()
  await expect(page.getByRole('heading', { name: 'Results Overview' })).toBeVisible({ timeout: 180000 })
  await page.getByRole('tab', { name: 'RF-SMOTE', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Fraud Threshold Line: 43.00%' })).toBeVisible()
  await expect(page.getByText('Ground Truth', { exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'See Full SHAP Evaluation' }).click()
  await expect(page.getByRole('dialog').locator('img')).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('live-receipt-shap.png') })
  await page.keyboard.press('Escape')
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download results' }).click()
  await (await download).saveAs(testInfo.outputPath('live-receipt.zip'))
  await page.getByRole('button', { name: 'Clear results' }).click()
  await expect(page).toHaveURL(/\/upload$/)
})
