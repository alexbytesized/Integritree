import { test, expect } from '@playwright/test'

const analysisKey = 'integritree-research-analysis'
const researchToken = 'integritree-research-session'
const receiptKey = 'integritree-receipt-id'
const receiptToken = 'integritree-receipt-session'

async function setup(page, kind, changes = {}) {
  const receipt = kind === 'receipt'
  const job = { id: 'error-fixture', filename: receipt ? 'unrecognized.png' : 'unsupported.csv',
    status: receipt ? 'unsupported' : 'failed', busy: false, revision: 0,
    error_code: receipt ? 'unsupported_receipt_layout' : 'unsupported_csv_layout',
    error: receipt ? 'The receipt layout is unclear or unsupported.' : 'Missing columns: amount. Unsupported columns: Action.',
    ...changes }
  const state = { job, deletes: [], deleteStatus: 204, gate: null, retries: 0 }
  if (receipt) await page.routeWebSocket('**/api/v1/receipts/sessions/presence', ws => {
    ws.onMessage(() => ws.send(JSON.stringify({ status: 'connected' })))
  })
  await page.route(`**/api/v1/${receipt ? 'receipts' : 'research'}/**`, async route => {
    const request = route.request()
    if (request.method() === 'DELETE') {
      state.deletes.push(new URL(request.url()).pathname)
      if (state.gate) await state.gate
      return state.deleteStatus === 204 ? route.fulfill({ status: 204 })
        : route.fulfill({ status: state.deleteStatus, json: { detail: 'Cleanup temporarily unavailable.' } })
    }
    if (request.url().endsWith('/retry')) {
      state.retries++
      Object.assign(job, { status: 'awaiting_confirmation', workflow: 'express_send', error: null,
        error_code: null, candidates: { amount: '100', date: '2026-10-05', time: '13:00' } })
      return route.fulfill({ status: 202, json: { id: job.id } })
    }
    if (request.url().endsWith('/image')) return route.fulfill({ status: 404 })
    return route.fulfill({ json: job })
  })
  await page.goto(receipt ? '/upload' : '/researcher-upload')
  await page.evaluate(({ key, token }) => {
    sessionStorage.setItem(key, 'error-fixture')
    sessionStorage.setItem(token, 'error-test-token')
  }, { key: receipt ? receiptKey : analysisKey, token: receipt ? receiptToken : researchToken })
  await page.goto(receipt ? '/results?receipt=error-fixture' : '/researcher?analysis=error-fixture')
  await expect(page.getByRole('heading', { name: 'Error Encountered' })).toBeVisible()
  return state
}

for (const kind of ['receipt', 'csv']) {
  test(`${kind} error wraps safely on desktop and mobile and exposes keyboard controls`, async ({ page }, testInfo) => {
    const filename = `${'long_filename_'.repeat(15)}<img src=x>.${kind === 'receipt' ? 'png' : 'csv'}`
    await setup(page, kind, { filename })
    const message = kind === 'receipt'
      ? `The receipt layout of the uploaded photo (${filename}) is unclear or unsupported. Use a complete GCash Express Send, Pay Online, or bank-account transfer screenshot.`
      : `The dataset layout of the uploaded CSV (${filename}) is unsupported. Ensure that all expected columns are present (amount, isFraud, nameDest, nameOrig, step, type).`
    await expect(page.getByRole('alert')).toHaveText(message)
    await expect(page.getByRole('alert').locator('img')).toHaveCount(0)
    await expect(page.locator('.screenshot-preview')).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Proceed' })).toHaveCount(0)
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }, { width: 320, height: 568 }]) {
      await page.setViewportSize(viewport)
      expect(await page.evaluate(() => document.body.scrollWidth <= innerWidth)).toBe(true)
      await expect(page.getByRole('alert')).toBeInViewport()
      await page.screenshot({ path: testInfo.outputPath(`${kind}-error-${viewport.width}.png`), fullPage: true })
      const clear = page.getByRole('button', { name: kind === 'receipt' ? 'Clear Receipt' : 'Clear CSV' })
      await clear.scrollIntoViewIfNeeded()
      await expect(clear).toBeInViewport()
      if (viewport.width === 320) await page.screenshot({ path: testInfo.outputPath(`${kind}-error-mobile-actions.png`) })
      await page.evaluate(() => { document.body.scrollTop = 0 })
    }
    await page.keyboard.press('Tab')
    await expect(page.getByRole(kind === 'receipt' ? 'button' : 'link', { name: 'Return', exact: true })).toBeFocused()
    await page.keyboard.press('Tab')
    if (kind === 'csv') {
      const summary = page.locator('summary')
      await expect(summary).toBeFocused()
      await expect(page.locator('details')).not.toHaveAttribute('open', '')
      await page.keyboard.press('Enter')
      await expect(page.getByText('Missing columns: amount. Unsupported columns: Action.', { exact: true })).toBeVisible()
      await page.keyboard.press('Tab')
    }
    await expect(page.getByRole('button', { name: kind === 'receipt' ? 'Clear Receipt' : 'Clear CSV' })).toBeFocused()
  })

  test(`${kind} cleanup disables repeat requests, retains failures, and clears the correct state`, async ({ page }) => {
    const state = await setup(page, kind)
    state.deleteStatus = 503
    let release
    state.gate = new Promise(resolve => { release = resolve })
    const clear = page.getByRole('button', { name: kind === 'receipt' ? 'Clear Receipt' : 'Clear CSV' })
    await clear.click()
    await expect(clear).toBeDisabled()
    if (kind === 'receipt') await expect(page.getByRole('button', { name: 'Return', exact: true })).toBeDisabled()
    await expect.poll(() => state.deletes.length).toBe(1)
    release()
    await expect(page.locator('.upload-error-page [role=alert]').last()).toHaveText('Cleanup temporarily unavailable.')
    await expect(clear).toBeEnabled()
    await expect(page.getByRole('heading', { name: 'Error Encountered' })).toBeVisible()
    expect(await page.evaluate(key => sessionStorage.getItem(key), kind === 'receipt' ? receiptKey : analysisKey)).toBe('error-fixture')
    state.deleteStatus = 204
    await clear.click()
    await expect(page).toHaveURL(kind === 'receipt' ? /\/upload$/ : /\/researcher-upload$/)
    expect(state.deletes).toEqual(Array(2).fill(kind === 'receipt' ? '/api/v1/receipts/sessions/current' : '/api/v1/research/analyses/error-fixture'))
    expect(await page.evaluate(key => sessionStorage.getItem(key), kind === 'receipt' ? receiptKey : analysisKey)).toBeNull()
    expect(await page.evaluate(key => sessionStorage.getItem(key), kind === 'receipt' ? receiptToken : researchToken)).toBe(kind === 'receipt' ? null : 'error-test-token')
  })
}

test('duplicate headers retain the friendly message and details; CSV Return preserves the analysis', async ({ page }, testInfo) => {
  const state = await setup(page, 'csv', { error: 'Duplicate CSV column headers are not allowed.' })
  await expect(page.getByRole('alert')).toContainText('The dataset layout of the uploaded CSV (unsupported.csv) is unsupported.')
  await page.locator('summary').click()
  await expect(page.locator('details')).toContainText('Duplicate CSV column headers are not allowed.')
  await page.screenshot({ path: testInfo.outputPath('csv-error-details-desktop.png') })
  await page.getByRole('link', { name: 'Return', exact: true }).click()
  await expect(page).toHaveURL(/researcher-upload$/)
  expect(state.deletes).toEqual([])
  expect(await page.evaluate(key => sessionStorage.getItem(key), analysisKey)).toBe('error-fixture')
})

test('invalid rows keep the specific error and readable, initially collapsed row issues', async ({ page }) => {
  await setup(page, 'csv', { error_code: null, error: 'Invalid predictor values.', issues: { step: { count: 3, rows: [2, 5, 8] } } })
  await expect(page.getByRole('alert')).toHaveText('Invalid predictor values.')
  const details = page.locator('details')
  await expect(details.locator('li')).not.toBeVisible()
  await page.locator('summary').click()
  await expect(details.locator('li')).toHaveText('step: 3 invalid value(s). Affected rows: 2, 5, 8.')
  await expect(details).toContainText('excluding the header')
  await expect(page.locator('pre')).toHaveCount(0)
})

for (const error of ['Only completed transaction receipts are supported.', 'Use a valid single PNG/JPEG, at most 10 MiB and 20 million pixels.']) {
  test(`receipt preserves specific rejection: ${error}`, async ({ page }) => {
    await setup(page, 'receipt', { error, error_code: null })
    await expect(page.getByRole('alert')).toHaveText(error)
    await expect(page.locator('.screenshot-preview')).toHaveCount(0)
    await page.getByRole('button', { name: 'Return', exact: true }).click()
    await expect(page).toHaveURL(/\/upload$/)
  })
}

test('receipt processing failure can retry and return to confirmation', async ({ page }) => {
  const state = await setup(page, 'receipt', { status: 'failed', error_code: null, error: 'Receipt processing failed. Retry processing.' })
  await expect(page.getByRole('alert')).toHaveText('Receipt processing failed. Retry processing.')
  await page.getByRole('button', { name: 'Retry processing' }).click()
  await expect(page.getByRole('heading', { name: 'Confirm Details' })).toBeVisible()
  expect(state.retries).toBe(1)
})

test('expired CSV analysis returns to upload and clears stale state', async ({ page }) => {
  await setup(page, 'csv')
  await page.route('**/api/v1/research/analyses/error-fixture', route => route.fulfill({ status: 410, json: { detail: 'Session expired.' } }))
  await page.reload()
  await expect(page).toHaveURL(/researcher-upload\?expired=1$/)
  expect(await page.evaluate(keys => keys.map(key => sessionStorage.getItem(key)), [analysisKey, researchToken])).toEqual([null, null])
})
