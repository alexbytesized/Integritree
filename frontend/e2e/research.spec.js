import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const backendPython = fileURLToPath(new URL('../../backend/.venv/Scripts/python.exe', import.meta.url))

// Allows validation against an isolated backend without restarting an active session.
test.beforeEach(async ({ page }) => {
  if (process.env.RESEARCH_API_BASE_URL) await page.route('**/api/v1/research/**', async route => {
    const url = new URL(route.request().url())
    const response = await route.fetch({ url: `${process.env.RESEARCH_API_BASE_URL}${url.pathname}${url.search}` })
    await route.fulfill({ response })
  })
})

const csv = 'step,type,amount,nameOrig,nameDest,isFraud\n' + Array.from({ length: 23 }, (_, i) =>
  `${i + 1},${i % 2 ? 'PAYMENT' : 'TRANSFER'},${100 + i * 1000},C_DEMO_${i},${i % 2 ? 'M' : 'C'}_DEMO,${i % 3 === 0 ? 1 : 0}\n`).join('')

test('upload, refresh, search, pagination, details, real SHAP, download and analyze another CSV', async ({ page }, testInfo) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/researcher-upload')
  await page.getByRole('button', { name: 'Upload Instructions' }).click()
  await expect(page.getByText('Required columns:', { exact: false })).toBeVisible()
  const template = page.waitForEvent('download')
  await page.getByRole('link', { name: 'Download Demonstration CSV Template' }).click()
  expect((await template).suggestedFilename()).toBe('research_template.csv')
  await page.getByRole('button', { name: 'Close dataset instructions' }).click()
  await page.locator('input[type=file]').setInputFiles({ name: 'browser_demo.csv', mimeType: 'text/csv', buffer: Buffer.from(csv) })
  await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
  await expect(page.getByRole('heading', { name: 'Results Overview' })).toBeVisible()
  await expect(page.getByText('browser_demo.csv', { exact: true })).toBeVisible()
  await expect(page.locator('.transactionTable tbody tr')).toHaveCount(10)
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('rfSmote')
  await page.getByRole('combobox', { name: 'Prediction outcome' }).selectOption('fp')
  await expect(page.locator('.transactionTable thead')).not.toContainText('Benchmark RF')
  await expect(page.getByText('Shared fraud threshold:', { exact: false })).toHaveCount(0)
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('both')
  await page.reload()
  await expect(page.getByRole('heading', { name: 'Results Overview' })).toBeVisible()
  await page.getByRole('button', { name: 'Next page' }).click()
  await expect(page.locator('.paginationRecords')).toContainText('11–20 of 23')
  await page.getByPlaceholder('Search Transaction ID').fill('23')
  await expect(page.locator('.transactionTable tbody tr')).toHaveCount(1)
  await expect(page.locator('.paginationRecords')).toContainText('1–1 of 1')
  await page.getByRole('button', { name: /^View transaction/ }).click()
  await expect(page.getByRole('heading', { name: 'Transaction Record', exact: true })).toBeVisible()
  await expect(page.locator('.td-field-value').filter({ hasText: /^C_DEMO_22$/ })).toBeVisible()
  await page.getByRole('tab', { name: 'RF-SMOTE', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'SHAP contribution bar chart' })).toHaveCount(0)
  await page.getByRole('button', { name: 'See Full SHAP Evaluation' }).click()
  await expect(page.getByRole('dialog', { name: 'RF-SMOTE — SHAP Evaluation' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'SHAP contribution bar chart' })).toHaveCount(0)
  await expect(page.locator('img[alt*="SHAP contribution bar chart"]')).toBeVisible()
  await expect(page.getByText(/Reconstruction error:/)).toHaveCount(0)
  await expect(page.locator('.shap-summary-sections')).toBeVisible()
  await page.locator('img[alt*="SHAP contribution bar chart"]').scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('details.png') })
  await page.locator('.shap-summary-sections section:last-child').scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('shap-footer.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ path: testInfo.outputPath('real-shap-mobile.png') })
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.getByRole('button', { name: 'Close SHAP evaluation' }).click()
  await page.getByRole('link', { name: 'Return', exact: true }).click()
  await expect(page.getByPlaceholder('Search Transaction ID')).toHaveValue('23')
  const downloadPromise = page.waitForEvent('download', { timeout: 150000 })
  await page.getByRole('button', { name: 'Download Results', exact: true }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe('integritree_results.zip')
  await download.saveAs(testInfo.outputPath('results.zip'))
  const exportedFiles = JSON.parse(execFileSync(backendPython, ['-c',
    'import json,re,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); names=[n for n in z.namelist() if re.fullmatch(r"Experiment-Paper_[0-9]{4}-[0-9]{2}-[0-9]{2}[.]pdf",n)]; assert len(names)==1; p=z.read(names[0]); assert p.startswith(b"%PDF-") and b"%%EOF" in p[-1024:]; print(json.dumps(z.namelist()))',
    testInfo.outputPath('results.zip')], { encoding: 'utf8' }))
  const paperName = exportedFiles.find(name => /^Experiment-Paper_\d{4}-\d{2}-\d{2}\.pdf$/.test(name))
  expect(exportedFiles.sort()).toEqual([
    paperName, paperName.replace('Experiment-Paper_', 'Raw-Data_').replace('.pdf', '.csv'),
  ])
  await page.getByRole('heading', { name: 'Results Overview' }).scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('results.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect(page.getByRole('button', { name: 'Download Results', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Download Results', exact: true }).scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('mobile.png') })
  const analysisId = new URL(page.url()).searchParams.get('analysis')
  await expect(page.getByRole('button', { name: 'Clear results', exact: true })).toHaveCount(0)
  await page.getByRole('link', { name: 'Clear Results', exact: true }).click()
  await expect(page).toHaveURL(/researcher-upload$/)
  // Clean up only this test's analysis, independently of the removed results action.
  await page.evaluate(async id => {
    const { api } = await import('/src/researchApi.js')
    const analysis = await api(`/analyses/${id}`)
    if (analysis.status !== 'complete') throw new Error('Navigation must retain the completed analysis')
    await api(`/analyses/${id}`, { method: 'DELETE' })
  }, analysisId)
  expect(errors).toEqual([])
})

test('invalid rows show actionable errors and allow retry', async ({ page }) => {
  await page.goto('/researcher-upload')
  await page.locator('input[type=file]').setInputFiles({ name: 'invalid.csv', mimeType: 'text/csv',
    buffer: Buffer.from('step,type,amount,nameOrig,nameDest,isFraud\n0,TRANSFER,10,C_A,C_B,1\n') })
  await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
  await expect(page.getByRole('alert')).toContainText('Invalid predictor values')
  await page.getByText('View details', { exact: true }).click()
  await expect(page.locator('.upload-error-details')).toContainText('step')
  await page.getByRole('link', { name: 'Return', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Upload File', exact: true })).toBeVisible()
})

test('an expired backend session returns to upload with an explanation', async ({ page }) => {
  await page.goto('/researcher?analysis=expired-session-test')
  await expect(page).toHaveURL(/researcher-upload\?expired=1$/)
  await expect(page.getByRole('alert')).toContainText('session expired')
})
