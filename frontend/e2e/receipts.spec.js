import { test, expect } from '@playwright/test'

const image = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64')
const fields = { workflow: 'express_send', category: 'TRANSFER', amount: '100.00', date: '2026-10-05', time: '13:00',
  reference: null, origin_role: 'client', destination_role: 'client' }
const derived = { hour_of_day: 13, day_of_week: 0, type_TRANSFER: 1, type_PAYMENT: 0, type_DEBIT: 0,
  type_CASH_IN: 0, type_CASH_OUT: 0, log_amount: 4.615, is_zero_amount: 0, is_merchant_origin: 0, is_merchant_dest: 0 }
const makeExplanation = revision => ({ status: 'computed', models: Object.fromEntries(['rf', 'rf_smote'].map(model => [model, {
  waterfall_url: `/api/v1/receipts/demo/waterfall/${model}?revision=${revision}`, chart_description: `${model} receipt contribution bar chart`,
  base_value: .1, output_value: .2, additivity_error: 0, features: [{ feature: 'hour_of_day', readable_value: 'Manila hour = 13', contribution: .1 }],
  top_positive_contributor: { status: 'available', feature: 'hour_of_day' },
}])) })

async function mockReceipts(page, unsupported = false, changes = {}) {
  await page.routeWebSocket('**/api/v1/receipts/sessions/presence', ws => {
    ws.onMessage(() => ws.send(JSON.stringify({ status: 'connected' })))
  })
  const job = { id: 'demo', filename: 'receipt.png', status: unsupported ? 'unsupported' : 'awaiting_confirmation',
    busy: false, revision: 0, workflow: unsupported ? undefined : 'express_send', candidates: { ...fields, date: null },
    error: unsupported ? 'The receipt layout is unclear or unsupported.' : null, ...changes }
  const confirmations = [], charts = [], deletions = []
  await page.route('**/api/v1/receipts**', async route => {
    const request = route.request(), url = new URL(request.url())
    const path = url.pathname
    if (path.endsWith('/sessions')) return route.fulfill({ json: { token: 'receipt-test' } })
    if (request.method() === 'DELETE') { deletions.push(path); return route.fulfill({ status: 204 }) }
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
  return { confirmations, charts, deletions, job }
}

async function begin(page, sample = null) {
  await page.goto('/upload')
  await page.locator('input[type=file]').setInputFiles(sample || { name: 'receipt.png', mimeType: 'image/png', buffer: image })
  await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
  await expect(page.getByRole('heading', { name: 'Confirm Details' })).toBeVisible()
}

async function confirm(page) {
  await page.getByLabel('Transaction Date:').fill('2026-10-05')
  await page.getByRole('button', { name: 'Proceed' }).click()
  await expect(page.getByRole('heading', { name: 'Results Overview' })).toBeVisible()
}

test('receipt confirmation, refresh, user model designs without labels, SHAP, revision and Clear', async ({ page }, testInfo) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const state = await mockReceipts(page)
  await begin(page)
  await page.getByRole('button', { name: 'Proceed' }).click()
  expect(state.confirmations).toHaveLength(0)
  await expect(page.getByLabel('Transaction Date:')).toHaveValue('')
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
    expect(state.charts.at(-1).url.searchParams.get('layout')).toBe('modal')
    await expect(page.locator('.shap-summary-sections li')).toHaveText(['10.00%', '20.00%', 'Hour of the Day'])
    expect(state.charts.at(-1).url.searchParams.has('presentation')).toBe(false)
    await page.keyboard.press('Escape')
  }
  await page.screenshot({ path: testInfo.outputPath('receipt-models-desktop.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('receipt-models-mobile.png'), fullPage: true })
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download Results' }).click()
  expect((await download).suggestedFilename()).toBe('integritree_receipt.zip')
  await page.getByRole('link', { name: 'Edit Details' }).click()
  await page.getByLabel('Transaction Amount:').fill('200.00')
  await confirm(page)
  expect(state.confirmations[1].expected_revision).toBe(1)
  expect(state.confirmations[1].fields.amount).toBe('200.00')
  await page.getByRole('button', { name: 'Clear Results' }).click()
  await expect(page).toHaveURL(/\/upload$/)
  expect(await page.evaluate(() => sessionStorage.getItem('integritree-receipt-id'))).toBeNull()
  expect(errors).toEqual([])
})

test('unsupported receipt cannot expose a confirmation form or fabricated results', async ({ page }) => {
  await mockReceipts(page, true)
  await begin(page)
  await expect(page.getByRole('alert')).toContainText('unsupported')
  await expect(page.getByRole('button', { name: 'Proceed' })).toHaveCount(0)
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
  await expect(page.getByLabel('Transaction Amount:')).not.toHaveValue('')
  // Keep actual extracted values; only complete a missing time/date for this UI check.
  if (!await page.getByLabel('Transaction Date:').inputValue()) await page.getByLabel('Transaction Date:').fill('2026-10-05')
  if (!await page.getByLabel('Transaction Time:').inputValue()) await page.getByLabel('Transaction Time:').fill('13:00')
  await page.getByRole('button', { name: 'Proceed' }).click()
  await expect(page.getByRole('heading', { name: 'Results Overview' })).toBeVisible({ timeout: 180000 })
  await page.getByRole('tab', { name: 'RF-SMOTE', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Fraud Threshold Line: 43.00%' })).toBeVisible()
  await expect(page.getByText('Ground Truth', { exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'See Full SHAP Evaluation' }).click()
  await expect(page.getByRole('dialog').locator('img')).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('live-receipt-shap.png') })
  await page.keyboard.press('Escape')
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download Results' }).click()
  await (await download).saveAs(testInfo.outputPath('live-receipt.zip'))
  await page.getByRole('button', { name: 'Clear Results' }).click()
  await expect(page).toHaveURL(/\/upload$/)
})


test('original form, all role defaults, overrides and drafts survive refresh', async ({ page }, testInfo) => {
  const state = await mockReceipts(page)
  await begin(page)
  await expect(page.locator('.details-row')).toHaveCount(7)
  await expect(page.locator('.particles-background canvas')).toBeVisible()
  await expect.poll(() => page.evaluate(() => window.pJSDom?.some(entry => entry.pJS.particles.array.length > 0))).toBe(true)
  expect(await page.locator('.particles-background canvas').evaluate(canvas => canvas.width > 0 && canvas.height > 0)).toBeTruthy()
  await expect(page.getByRole('checkbox')).toHaveCount(0)
  await expect(page.getByText(/Temporary storage:/)).toHaveCount(0)
  for (const [category, origin, destination] of [
    ['PAYMENT', 'client', 'merchant'], ['TRANSFER', 'client', 'client'],
    ['DEBIT', 'client', 'merchant'], ['CASH_IN', 'merchant', 'client'], ['CASH_OUT', 'client', 'merchant'],
  ]) {
    await page.getByLabel('Transaction Type:').selectOption(category)
    await expect(page.getByLabel('Sender Type:')).toHaveValue(origin)
    await expect(page.getByLabel('Recipient Type:')).toHaveValue(destination)
  }
  await page.getByLabel('Sender Type:').selectOption('merchant')
  await page.getByLabel('Recipient Type:').selectOption('client')
  await page.getByLabel('Transaction ID:').fill('0000123')
  await page.getByLabel('Transaction Amount:').fill('123.45')
  await page.getByLabel('Transaction Date:').fill('2026-10-05')
  await page.getByLabel('Transaction Time:').fill('23:59')
  await page.reload()
  await expect(page.getByLabel('Transaction Type:')).toHaveValue('CASH_OUT')
  await expect(page.getByLabel('Sender Type:')).toHaveValue('merchant')
  await expect(page.getByLabel('Recipient Type:')).toHaveValue('client')
  await expect(page.getByLabel('Transaction ID:')).toHaveValue('0000123')
  await expect(page.getByLabel('Transaction Amount:')).toHaveValue('123.45')
  await expect(page.getByLabel('Transaction Time:')).toHaveValue('23:59')
  await page.locator('.receipt-confirmation').scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('receipt-form-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.setViewportSize({ width: 390, height: 1400 })
  await page.locator('.receipt-confirmation').scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('receipt-form-mobile.png') })
  await page.getByRole('button', { name: 'Proceed' }).click()
  await expect(page.getByRole('heading', { name: 'Results Overview' })).toBeVisible()
  expect(state.confirmations[0].fields).toMatchObject({ category: 'CASH_OUT', origin_role: 'merchant', destination_role: 'client', reference: '0000123' })
  expect(state.confirmations[0].confirmed).toBe(true)
  await page.getByRole('link', { name: 'Return', exact: true }).click()
  await expect(page.getByLabel('Sender Type:')).toHaveValue('merchant')
  expect(state.deletions).toHaveLength(0)
  await page.getByRole('button', { name: 'Return', exact: true }).click()
  await expect(page).toHaveURL(/\/upload$/)
  expect(state.deletions).toEqual(['/api/v1/receipts/sessions/current'])
  expect(await page.evaluate(() => Object.keys(sessionStorage).filter(key => key.startsWith('integritree-receipt')))).toEqual([])
  await begin(page)
})

test('browser Back out of receipt clears its session', async ({ page }) => {
  const state = await mockReceipts(page)
  await begin(page)
  await page.goBack()
  await expect(page).toHaveURL(/\/upload$/)
  await expect.poll(() => state.deletions.length).toBe(1)
  expect(await page.evaluate(() => sessionStorage.getItem('integritree-receipt-session'))).toBeNull()
})


test('failed Return cleanup stays retryable and a late confirmation cannot reopen results', async ({ page }) => {
  await mockReceipts(page)
  await begin(page)
  let failures = 1
  await page.route('**/api/v1/receipts/sessions/current', route => {
    if (failures-- > 0) return route.fulfill({ status: 503, json: { detail: 'Cleanup temporarily unavailable.' } })
    return route.fulfill({ status: 204 })
  })
  await page.getByRole('button', { name: 'Return', exact: true }).click()
  await expect(page).toHaveURL(/\/results\?receipt=demo$/)
  await expect(page.getByRole('button', { name: 'Retry cleanup' })).toBeVisible()
  expect(await page.evaluate(() => sessionStorage.getItem('integritree-receipt-id'))).toBe('demo')
  await page.getByRole('button', { name: 'Return', exact: true }).click()
  await expect(page).toHaveURL(/\/upload$/)
  await begin(page)
  let release
  const gate = new Promise(resolve => { release = resolve })
  await page.route('**/api/v1/receipts/demo/confirm', async route => {
    await gate
    await route.fulfill({ status: 202, json: { id: 'demo', revision: 1, status: 'predicting' } })
  })
  await page.getByLabel('Transaction Date:').fill('2026-10-05')
  const confirming = page.waitForRequest('**/demo/confirm')
  await page.getByRole('button', { name: 'Proceed' }).click()
  await confirming
  await page.getByRole('button', { name: 'Return', exact: true }).click()
  await expect(page).toHaveURL(/\/upload$/)
  const response = page.waitForResponse('**/demo/confirm')
  release()
  await response
  await expect(page).toHaveURL(/\/upload$/)
})


test('receipt loading reuses researcher skeleton and preserves clear and edit navigation', async ({ page }, testInfo) => {
  const state = await mockReceipts(page, false, { busy: true, status: 'extracting' })
  await page.goto('/upload')
  await page.locator('input[type=file]').setInputFiles({ name: 'receipt.png', mimeType: 'image/png', buffer: image })
  await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
  await expect(page.getByRole('status')).toHaveText('Reading your receipt...')
  await expect(page.locator('.research-loading-spinner')).toBeVisible()
  await expect(page.locator('.research-loading-skeleton')).toBeVisible()
  await expect(page.locator('.screenshot-placeholder')).toHaveCount(0)
  await page.screenshot({ path: testInfo.outputPath('receipt-reading.png') })
  await page.getByRole('button', { name: 'Return to upload' }).click()
  await expect(page).toHaveURL(/\/upload$/)
  expect(state.deletions).toHaveLength(1)
  Object.assign(state.job, { busy: false, status: 'awaiting_confirmation' })
  await begin(page)
  await page.getByLabel('Transaction Date:').fill('2026-10-05')
  await page.route('**/demo/confirm', route => {
    Object.assign(state.job, { busy: true, status: 'predicting' })
    return route.fulfill({ status: 202, json: { id: 'demo', revision: 1 } })
  })
  await page.getByRole('button', { name: 'Proceed' }).click()
  await expect(page.getByRole('status')).toHaveText('Preparing your predictions and explanations...')
  await expect(page.locator('.research-loading-spinner')).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('receipt-predicting.png') })
  await page.getByRole('link', { name: 'Return to receipt' }).click()
  await expect(page).toHaveURL(/\/results\?receipt=demo$/)
  expect(state.deletions).toHaveLength(1)
})

test('receipt controls, amount tooltip, minute time and particle sizing', async ({ page }, testInfo) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const state = await mockReceipts(page, false, { candidates: { ...fields, time: '08:47:32' } })
  await begin(page)
  await expect(page.getByLabel('Transaction Time:')).toHaveValue('08:47')
  await expect(page.getByLabel('Transaction Time:')).toHaveAttribute('step', '60')
  const amount = page.getByLabel('Transaction Amount:', { exact: true })
  await amount.fill('')
  await amount.pressSequentially('abc-+1e2.34.5')
  await expect(amount).toHaveValue('12.34')
  await amount.fill('99.99')
  await amount.fill('100')
  await expect(page.getByText(/Check each value|Experimental predictions/)).toHaveCount(0)
  const clear = page.getByRole('button', { name: 'Clear Receipt', exact: true })
  const proceed = page.getByRole('button', { name: 'Proceed', exact: true })
  const actions = page.locator('.receipt-form-actions')
  await expect(actions.getByRole('button')).toHaveText(['Proceed', 'Clear Receipt'])
  await actions.scrollIntoViewIfNeeded()
  await expect(clear).toHaveCSS('background-color', 'rgb(255, 255, 255)')
  await expect(clear).toHaveCSS('border-top-color', 'rgb(8, 109, 193)')
  await clear.hover()
  await expect(clear).toHaveCSS('background-color', 'rgb(8, 109, 193)')
  await proceed.hover()
  await expect(proceed).toHaveCSS('background-color', 'rgb(255, 255, 255)')
  await expect(proceed).toHaveCSS('border-top-color', 'rgb(8, 109, 193)')
  const help = page.getByRole('button', { name: 'Transaction Amount Details' })
  await help.hover()
  await expect(page.getByRole('tooltip')).toContainText('Transaction amount refers to the PHP principal amount (excluding fees)')
  await page.screenshot({ path: testInfo.outputPath('amount-tooltip-desktop.png') })
  await page.keyboard.press('Escape')
  await expect(page.getByRole('tooltip')).toHaveCount(0)
  await help.focus()
  await expect(page.getByRole('tooltip')).toBeVisible()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('tooltip')).toHaveCount(0)
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1100 })
    await expect.poll(() => page.evaluate(() => {
      const canvas = document.querySelector('.particles-background canvas')
      if (!canvas) return false
      const bounds = canvas.getBoundingClientRect()
      return Math.abs(canvas.width / bounds.width - canvas.height / bounds.height) < .01
    })).toBe(true)
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
    await page.locator('.receipt-confirmation').scrollIntoViewIfNeeded()
    if (width === 390) { await help.click(); await expect(page.getByRole('tooltip')).toBeVisible() }
    await page.screenshot({ path: testInfo.outputPath(`receipt-controls-${width}.png`) })
    await page.keyboard.press('Escape')
  }
  await confirm(page)
  expect(state.confirmations[0].fields.time).toBe('08:47')
  expect(state.confirmations[0].fields.amount).toBe('100')
  const original = page.locator('.td-card').first()
  await expect(original.locator('.td-field-label')).toHaveText([
    'Transaction ID:', 'Transaction Amount:', 'Transaction Type:', 'Transaction Date:',
    'Transaction Time:', 'Sender Type:', 'Recipient Type:',
  ])
  await expect(original.locator('.td-field-value').nth(3)).toHaveText('05/10/2026')
  await expect(original.locator('.td-field-value').nth(4)).toHaveText('08:47 am')
  await original.getByRole('button', { name: 'Transaction Amount Details' }).click()
  await expect(page.getByRole('tooltip')).toContainText('Transaction amount refers to the PHP principal amount (excluding fees)')
  await expect(page.getByRole('tooltip').locator('p')).toHaveCSS('text-align', 'justify')
  await page.screenshot({ path: testInfo.outputPath('results-amount-tooltip-mobile.png') })
  await page.keyboard.press('Escape')
  expect(errors).toEqual([])
})

test('all four receipt result tabs share heading, transaction ID and bottom actions', async ({ page }, testInfo) => {
  await mockReceipts(page)
  await begin(page)
  await confirm(page)
  const title = page.getByRole('heading', { name: 'Results Overview' })
  await expect(title).toHaveCSS('color', 'rgb(35, 39, 42)')
  await title.scrollIntoViewIfNeeded()
  const titleBox = await title.boundingBox()
  const backBox = await page.getByRole('link', { name: 'Return', exact: true }).boundingBox()
  expect(titleBox.y).toBeGreaterThan(backBox.y + backBox.height + 15)
  await expect(page.getByText(/Experimental transaction predictions|Download any results you want to keep/)).toHaveCount(0)
  await page.screenshot({ path: testInfo.outputPath('receipt-overview.png') })
  const footer = page.locator('footer.receipt-results-actions')
  for (const tab of ['Transaction Details', 'RF-SMOTE', 'Benchmark RF', 'Both Models']) {
    await page.getByRole('tab', { name: tab, exact: true }).click()
    await expect(page.locator('.td-record-id')).toHaveText('Transaction ID: Not provided')
    await expect(footer.locator('button, a')).toHaveText(['Download Results', 'Edit Details', 'Clear Results'])
    const contentBottom = await page.locator('.td-content-section').evaluate(node => node.getBoundingClientRect().bottom)
    const footerTop = await footer.evaluate(node => node.getBoundingClientRect().top)
    expect(footerTop).toBeGreaterThan(contentBottom)
    for (const name of ['Download Results', 'Edit Details', 'Clear Results']) {
      const control = footer.getByRole(name === 'Edit Details' ? 'link' : 'button', { name, exact: true })
      await page.mouse.move(0, 0)
      await expect(control).toHaveCSS('background-color', 'rgb(255, 255, 255)')
      await control.hover()
      await expect(control).toHaveCSS('background-color', 'rgb(8, 109, 193)')
      await expect(control).toHaveCSS('color', 'rgb(255, 255, 255)')
    }
    await footer.scrollIntoViewIfNeeded()
    await page.screenshot({ path: testInfo.outputPath(`receipt-footer-${tab.replaceAll(' ', '-')}.png`) })
  }
  await page.setViewportSize({ width: 390, height: 844 })
  for (const tab of ['Transaction Details', 'RF-SMOTE', 'Benchmark RF', 'Both Models']) {
    await page.getByRole('tab', { name: tab, exact: true }).click()
    await footer.scrollIntoViewIfNeeded()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
    for (const control of await footer.locator('button, a').all()) await expect(control).toBeInViewport()
    await page.screenshot({ path: testInfo.outputPath(`receipt-mobile-footer-${tab.replaceAll(' ', '-')}.png`) })
  }
  await footer.getByRole('link', { name: 'Edit Details' }).click()
  await page.getByLabel('Transaction ID:').fill('00001234')
  await confirm(page)
  await expect(page.locator('.td-record-id')).toHaveText('Transaction ID: 00001234')
  await footer.getByRole('button', { name: 'Clear Results' }).click()
  await expect(page).toHaveURL(/\/upload$/)
})
