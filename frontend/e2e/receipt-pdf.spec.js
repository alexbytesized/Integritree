import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

// Uses receipt_browser_app.py: synthetic OCR/models, real export and presence.
test.skip(!process.env.RECEIPT_LIFECYCLE_LIVE, 'Opt-in synthetic HTTP/WebSocket backend')
const backendPython = fileURLToPath(new URL('../../backend/.venv/Scripts/python.exe', import.meta.url))
const image = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64')

test('real receipt ZIP includes both landscape summaries from either model tab', async ({ page, request }, testInfo) => {
  await page.goto('/upload')
  await page.locator('input[type=file]').setInputFiles({ name: 'synthetic.png', mimeType: 'image/png', buffer: image })
  await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
  await expect(page.getByLabel('Transaction Amount:')).toHaveValue('100.00')
  await page.getByLabel('Transaction ID:').fill('00007')
  await page.getByRole('button', { name: 'Proceed' }).click()
  await expect(page.getByRole('heading', { name: 'Results Overview' })).toBeVisible()
  const receipt = await page.evaluate(() => ({ token: sessionStorage.getItem('integritree-receipt-session'), id: sessionStorage.getItem('integritree-receipt-id') }))

  let failFirstDownload = true
  await page.route('**/api/v1/receipts/*/download?revision=*', route => {
    if (!failFirstDownload) return route.continue()
    failFirstDownload = false
    return route.fulfill({ status: 503, json: { detail: 'The transaction PDF could not be generated. Try Download Results again.' } })
  })
  await page.getByRole('button', { name: 'Download Results' }).click()
  await expect(page.getByRole('alert')).toContainText('Try Download Results again')

  for (const tab of ['Benchmark RF', 'RF-SMOTE']) {
    await page.getByRole('tab', { name: tab, exact: true }).click()
    const downloading = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Download Results' }).click()
    const download = await downloading
    expect(download.suggestedFilename()).toBe('integritree_receipt.zip')
    const zipPath = testInfo.outputPath(`${tab}.zip`)
    await download.saveAs(zipPath)
    const exported = JSON.parse(execFileSync(backendPython, ['-c', `
import io,json,re,sys,zipfile
from pypdf import PdfReader
with zipfile.ZipFile(sys.argv[1]) as archive:
    names = archive.namelist()
    pdfs = [n for n in names if re.fullmatch(r'Transaction-Results_[0-9]{4}-[0-9]{2}-[0-9]{2}[.]pdf', n)]
    assert len(pdfs) == 1
    pdf = PdfReader(io.BytesIO(archive.read(pdfs[0])))
    print(json.dumps({'files': names, 'pages': [{'width': float(p.mediabox.width), 'height': float(p.mediabox.height), 'text': p.extract_text()} for p in pdf.pages]}))
`, zipPath], { encoding: 'utf8' }))
    expect(exported.files).toHaveLength(7)
    expect(exported.files).toEqual(expect.arrayContaining(['confirmed_inputs.json', 'results.json', 'explanations.json', 'metadata.json', 'rf_shap_contributions.svg', 'rf_smote_shap_contributions.svg']))
    expect(exported.pages).toHaveLength(2)
    for (const [index, model] of ['Benchmark RF', 'RF-SMOTE'].entries()) {
      expect(exported.pages[index].width).toBeGreaterThan(exported.pages[index].height)
      expect(exported.pages[index].text).toContain(`Transaction Record · ${model}`)
      expect(exported.pages[index].text).toContain('Transaction ID: 00007')
      expect(exported.pages[index].text).toContain('SHAP Explanation')
      expect(exported.pages[index].text.replace(/\s+/g, ' ')).toContain('Manila hour: 13 (+10.00 percentage points)')
    }
  }
  await page.getByRole('button', { name: 'Clear Results' }).click()
  await expect(page).toHaveURL(/\/upload$/)
  const cleared = await request.get(`/api/v1/receipts/${receipt.id}/download?revision=1`, { headers: { 'X-Receipt-Session': receipt.token } })
  expect(cleared.status()).toBe(410)
})
