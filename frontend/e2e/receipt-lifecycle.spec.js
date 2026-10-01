import { test, expect } from '@playwright/test'

// Run the isolated backend/tests/receipt_browser_app.py server on port 8001 and Vite on 5174 (INTEGRITREE_API_TARGET).
// HTTP and WebSocket both travel through Vite; these tests never mock cleanup.
test.skip(!process.env.RECEIPT_LIFECYCLE_LIVE, 'Opt-in synthetic HTTP/WebSocket backend')
const image = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64')
async function upload(page) {
  await page.goto('/upload')
  await page.locator('input[type=file]').setInputFiles({ name: 'synthetic.png', mimeType: 'image/png', buffer: image })
  await page.getByRole('button', { name: 'Extract transaction details' }).click()
  await expect(page.getByLabel('Transaction Amount:')).toHaveValue('100.00')
  return page.evaluate(() => ({ token: sessionStorage.getItem('integritree-receipt-session'), id: sessionStorage.getItem('integritree-receipt-id') }))
}
const status = (request, receipt) => request.get(`/api/v1/receipts/${receipt.id}`, { headers: { 'X-Receipt-Session': receipt.token } })

test('real refresh, background tab, results/edit, close grace and session isolation', async ({ page, context, request }) => {
  const receipt = await upload(page)
  await page.getByLabel('Transaction ID:').fill('00007')
  await page.reload()
  await expect(page.getByLabel('Transaction ID:')).toHaveValue('00007')
  await page.getByRole('button', { name: 'Proceed' }).click()
  await expect(page.getByRole('heading', { name: 'Results Overview' })).toBeVisible()
  await page.getByRole('link', { name: 'Return', exact: true }).click()
  await expect(page.getByLabel('Transaction ID:')).toHaveValue('00007')
  const other = await context.newPage()
  const independent = await upload(other)
  expect(independent.token).not.toBe(receipt.token)
  await other.bringToFront()
  expect((await status(request, receipt)).status()).toBe(200)
  await page.close()
  expect((await status(request, receipt)).status()).toBe(200)
  await expect.poll(async () => (await status(request, receipt)).status(), { timeout: 45000, intervals: [1000] }).toBe(410)
  expect((await status(request, independent)).status()).toBe(200)
  await other.getByRole('button', { name: 'Return', exact: true }).click()
  await expect(other).toHaveURL(/\/upload$/)
  expect((await status(request, independent)).status()).toBe(410)
  await other.close()
})

test('real Back cleanup permits another upload and stale history cannot restore it', async ({ page, request }) => {
  const receipt = await upload(page)
  await page.goBack()
  await expect(page).toHaveURL(/\/upload$/)
  await expect.poll(async () => (await status(request, receipt)).status()).toBe(410)
  await page.goForward()
  await expect(page).toHaveURL(/\/upload$/)
  const replacement = await upload(page)
  expect(replacement.token).not.toBe(receipt.token)
  await page.getByRole('button', { name: 'Return', exact: true }).click()
})
