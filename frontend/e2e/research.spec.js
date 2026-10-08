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
  await page.getByRole('link', { name: 'Download Raw CSV Example' }).click()
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
    'import csv,io,json,re,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); names=[n for n in z.namelist() if re.fullmatch(r"Experiment-Paper_[0-9]{4}-[0-9]{2}-[0-9]{2}[.]pdf",n)]; assert len(names)==1; p=z.read(names[0]); assert p.startswith(b"%PDF-") and b"%%EOF" in p[-1024:]; n=next(n for n in z.namelist() if n.startswith("Raw-Results_")); rows=list(csv.DictReader(io.StringIO(z.read(n).decode()))); assert len(rows)==23; assert list(rows[0])==["transaction_id","upload_row_number","step","type","amount","nameOrig","nameDest","isFraud","rf_score","rf_predicted_label","rf_smote_score","rf_smote_predicted_label"]; assert all(None not in r and None not in r.values() for r in rows); print(json.dumps(z.namelist()))',
    testInfo.outputPath('results.zip')], { encoding: 'utf8' }))
  const paperName = exportedFiles.find(name => /^Experiment-Paper_\d{4}-\d{2}-\d{2}\.pdf$/.test(name))
  expect(exportedFiles.sort()).toEqual([
    paperName, paperName.replace('Experiment-Paper_', 'Raw-Results_').replace('.pdf', '.csv'),
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

test('prepared CSV upload supports details, real SHAP and complete export', async ({ page }, testInfo) => {
  const prepared = execFileSync(backendPython, ['-c',
    'import io,sys,pandas as pd; from integritree.settings import load_settings; from integritree.ml.preprocessing import FittedPreprocessor; from integritree.ml.features import SOURCE_COLUMNS; s=load_settings(); raw=pd.read_csv(io.StringIO(sys.stdin.read())); p=FittedPreprocessor.load(s.research_prepared_dir / "preprocessing.json"); x=p.transform(raw[SOURCE_COLUMNS]); x["isFraud"]=raw.isFraud; sys.stdout.write(x.loc[:,list(reversed(x.columns))].to_csv(index=False,lineterminator="\\n"))',
  ], { input: csv, encoding: 'utf8' })
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/researcher-upload')
  await page.locator('input[type=file]').setInputFiles({ name: 'prepared_demo.csv', mimeType: 'text/csv', buffer: Buffer.from(prepared) })
  await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
  await expect(page.locator('.transactionTable tbody tr')).toHaveCount(10)
  const analysisId = new URL(page.url()).searchParams.get('analysis')
  await page.getByRole('button', { name: /^View transaction/ }).first().click()
  await expect(page.getByText('N/A (file uploaded is already preprocessed)', { exact: true })).toHaveCount(9)
  await expect(page.locator('.td-field-row').filter({ hasText: 'Transaction is Fraudulent:' })).toContainText('1 (True)')
  await expect(page.locator('.td-field-row').filter({ hasText: 'Day of the Week' })).toContainText('0 (Monday)')
  await page.getByRole('tab', { name: 'RF-SMOTE', exact: true }).click()
  await page.getByRole('button', { name: 'See Full SHAP Evaluation' }).click()
  await expect(page.getByRole('dialog').locator('img')).toBeVisible()
  await page.getByRole('button', { name: 'Close SHAP evaluation' }).click()
  await page.getByRole('link', { name: 'Return', exact: true }).click()
  const downloading = page.waitForEvent('download', { timeout: 150000 })
  await page.getByRole('button', { name: 'Download Results', exact: true }).click()
  const download = await downloading
  await download.saveAs(testInfo.outputPath('prepared-results.zip'))
  const exported = JSON.parse(execFileSync(backendPython, ['-c',
    'import csv,io,json,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); n=next(n for n in z.namelist() if n.startswith("Raw-Results_")); rows=list(csv.DictReader(io.StringIO(z.read(n).decode()))); print(json.dumps({"rows":len(rows),"first":rows[0]}))',
    testInfo.outputPath('prepared-results.zip'),
  ], { encoding: 'utf8' }))
  expect(exported.rows).toBe(23)
  expect(exported.first.isFraud).toBe('1')
  expect(exported.first).toHaveProperty('log_amount')
  expect(exported.first).not.toHaveProperty('step')
  for (const column of ['explanation_status', 'rf_narrative', 'rf_smote_narrative']) {
    expect(exported.first).not.toHaveProperty(column)
  }
  for (const column of ['transaction_id', 'upload_row_number', 'rf_score', 'rf_predicted_label', 'rf_smote_score', 'rf_smote_predicted_label']) {
    expect(exported.first).toHaveProperty(column)
  }
  await page.evaluate(async id => {
    const { api } = await import('/src/researchApi.js')
    const job = await api(`/analyses/${id}`)
    if (job.input_format !== 'prepared' || job.export.metadata.preprocessing_applied !== false) throw new Error('Incorrect prepared-upload metadata')
  }, analysisId)
  await page.getByRole('link', { name: 'Clear Results', exact: true }).click()
  await expect(page).toHaveURL(/researcher-upload$/)
  await expect.poll(() => page.evaluate(async id => {
    const { api } = await import('/src/researchApi.js')
    try {
      await api(`/analyses/${id}`, { method: 'DELETE' })
      return 204
    } catch (error) {
      if (error.status === 409) return 409
      throw error
    }
  }, analysisId), { timeout: 60000 }).toBe(204)
  expect(errors).toEqual([])
})

test('an expired backend session returns to the upload page without an alert', async ({ page }) => {
  await page.goto('/researcher?analysis=expired-session-test')
  await expect(page).toHaveURL(/researcher-upload\?expired=1$/)
  await expect(page.getByRole('heading', { name: 'Upload File', exact: true })).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
})
