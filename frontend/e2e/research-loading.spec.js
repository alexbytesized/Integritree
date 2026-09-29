import { test, expect } from '@playwright/test'

const analysisUrl = '/researcher?analysis=loading-fixture'

function completedJob() {
  const metrics = Object.fromEntries(['precision', 'recall', 'f1', 'mcc', 'pr_auc'].map(key => [key, { value: 1 }]))
  const model = { metrics, actual_fraud: 1, confusion_matrix: { tp: 1, tn: 2, fp: 0, fn: 0 } }
  return { status: 'complete', filename: 'loading-fixture.csv', rows_processed: 3, threshold: .43,
    evaluation: { models: { rf: model, rf_smote: model }, descriptive_comparisons: metrics,
      statistical_test: { p_value: 1, table: { both_correct: 3, rf_only_correct: 0, smote_only_correct: 0, both_incorrect: 0 } } } }
}

test('loading follows real stages and counts, survives refresh, then shows results', async ({ page }) => {
  let job = { status: 'uploading', rows_processed: 0 }
  let releaseFirstRequest
  const firstRequest = new Promise(resolve => { releaseFirstRequest = resolve })
  await page.route('**/api/v1/research/analyses/loading-fixture**', async route => {
    if (route.request().url().includes('/records')) return route.fulfill({ json: { records: [], total: 0, page: 1 } })
    await firstRequest
    await route.fulfill({ json: job })
  })
  await page.goto(analysisUrl)
  const status = page.getByRole('status')
  await expect(status).toHaveText('Connecting to your analysis…')
  await expect(page.locator('.research-loading-count')).toHaveCount(0)
  releaseFirstRequest()
  for (const [state, message] of [
    ['uploading', 'Uploading your CSV…'],
    ['queued', 'Waiting to start your analysis…'],
    ['loading_models', 'Verifying fraud detection models…'],
    ['processing', 'Analyzing your transaction records…'],
    ['evaluating', 'Calculating your analysis results…'],
  ]) {
    job = { status: state, rows_processed: 1250 }
    await expect(status).toHaveText(message)
    if (['processing', 'evaluating'].includes(state)) {
      await expect(page.locator('.research-loading-count')).toHaveText('1,250 records processed.')
      await expect(status).not.toContainText('records processed')
    } else await expect(page.locator('.research-loading-count')).toHaveCount(0)
    if (state === 'processing') {
      job.rows_processed = 2500
      await expect(page.locator('.research-loading-count')).toHaveText('2,500 records processed.')
      await expect(status).toHaveText(message)
      await page.reload()
      await expect(status).toHaveText(message)
      await expect(page.locator('.research-loading-count')).toHaveText('2,500 records processed.')
    }
  }
  job = completedJob()
  await expect(page.getByRole('heading', { name: 'Results Overview', exact: true })).toBeVisible()
  await expect(page.locator('.research-loading')).toHaveCount(0)
  await expect(page.getByText('loading-fixture.csv', { exact: true })).toBeVisible()
})

for (const failure of ['analysis', 'connection']) {
  test(`${failure} errors stop loading and retain recovery controls`, async ({ page }) => {
    let failed = false
    await page.route('**/api/v1/research/analyses/loading-fixture', route => {
      if (!failed) return route.fulfill({ json: { status: 'processing', rows_processed: 12 } })
      return failure === 'analysis'
        ? route.fulfill({ json: { status: 'failed', error: 'Invalid predictor values.', issues: { step: { rows: [13] } } } })
        : route.fulfill({ status: 503, json: { detail: 'The analysis service is unavailable.' } })
    })
    await page.goto(analysisUrl)
    await expect(page.getByRole('status')).toHaveText('Analyzing your transaction records…')
    failed = true
    await expect(page.getByRole('alert')).toContainText(failure === 'analysis' ? 'Invalid predictor values.' : 'The analysis service is unavailable.')
    await expect(page.locator('.research-loading')).toHaveCount(0)
    if (failure === 'analysis') {
      await expect(page.locator('pre')).toContainText('13')
      await expect(page.getByRole('button', { name: 'Clear failed analysis' })).toBeVisible()
    }
    await page.getByRole('link', { name: 'Return to upload / retry' }).click()
    await expect(page.getByRole('heading', { name: 'Upload File', exact: true })).toBeVisible()
  })
}

test('desktop and mobile skeletons keep the status centered and honor reduced motion', async ({ page }, testInfo) => {
  let deleted = false
  await page.route('**/api/v1/research/analyses/loading-fixture', route => {
    if (route.request().method() === 'DELETE') deleted = true
    return route.fulfill({ json: { status: 'processing', rows_processed: 1250 } })
  })
  await page.goto(analysisUrl)
  await expect(page.getByRole('status')).toHaveText('Analyzing your transaction records…')
  await expect(page.locator('.research-loading-skeleton')).toHaveAttribute('aria-hidden', 'true')
  await expect(page.locator('.research-loading-spinner')).toHaveAttribute('aria-hidden', 'true')
  await expect(page.locator('.research-loading-table-row')).toHaveCount(6)
  await expect(page.getByRole('status')).toHaveAttribute('aria-live', 'polite')
  await expect(page.getByRole('status')).toHaveAttribute('aria-atomic', 'true')
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }, { width: 320, height: 568 }]) {
    await page.setViewportSize(viewport)
    await expect(page.locator('.research-loading-spinner')).toHaveCSS('width', '56px')
    await expect(page.locator('.research-loading-spinner')).toHaveCSS('height', '56px')
    const box = await page.locator('.research-loading-status').boundingBox()
    expect(Math.abs(box.x + box.width / 2 - viewport.width / 2)).toBeLessThan(2)
    expect(Math.abs(box.y + box.height / 2 - viewport.height / 2)).toBeLessThan(2)
    expect(await page.evaluate(() => document.body.scrollWidth <= window.innerWidth)).toBe(true)
    await page.screenshot({ path: testInfo.outputPath(`loading-${viewport.width}.png`) })
  }
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await expect(page.locator('.research-loading-spinner')).toHaveCSS('animation-name', 'none')
  await expect(page.locator('.research-loading-placeholder').first()).toHaveCSS('animation-name', 'none')
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: 'Return to upload', exact: true })).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page).toHaveURL(/researcher-upload$/)
  expect(deleted).toBe(false)
})

test('a missing analysis keeps the upload prompt instead of a perpetual loader', async ({ page }) => {
  await page.goto('/researcher')
  await expect(page.getByText('Upload a CSV to start a new analysis.')).toBeVisible()
  await expect(page.locator('.research-loading')).toHaveCount(0)
})
