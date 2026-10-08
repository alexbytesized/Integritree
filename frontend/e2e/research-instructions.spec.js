import { test, expect } from '@playwright/test'

async function openGuide(page) {
  await page.goto('/researcher-upload')
  await page.getByRole('button', { name: 'Upload Instructions' }).click()
  return page.getByRole('dialog', { name: 'CSV Upload Guide' })
}

test('guide separates raw and prepared instructions and keeps the existing download', async ({ page }) => {
  const template = 'step,type,amount,nameOrig,nameDest,isFraud\n1,TRANSFER,100,C_DEMO_ORIGIN,C_DEMO_DESTINATION,0\n'
  await page.route('**/api/v1/research/template', route => route.fulfill({
    contentType: 'text/csv', body: template,
    headers: { 'Content-Disposition': 'attachment; filename="research_template.csv"' },
  }))
  const dialog = await openGuide(page)
  await expect(dialog.getByRole('heading', { level: 3 })).toHaveText([
    'Find your file format', 'File requirements', 'Need an example?',
  ])
  await expect(dialog.getByRole('tab', { name: 'Raw CSV', exact: true })).toHaveAttribute('aria-selected', 'true')
  const raw = dialog.getByRole('tabpanel', { name: 'Raw CSV', exact: true })
  await expect(raw.getByRole('list', { name: 'Raw CSV required columns' }).locator('li')).toHaveText([
    'step', 'type', 'amount', 'nameOrig', 'nameDest', 'isFraud',
  ])
  await expect(raw.locator('details')).not.toHaveAttribute('open', '')
  await raw.locator('summary').click()
  await expect(raw).toContainText('A blank amount is filled using the saved training median.')
  await expect(raw.locator('details li').last()).toHaveText('isFraud must be 0 for legitimate or 1 for fraudulent.')
  await dialog.getByRole('tab', { name: 'Preprocessed CSV' }).click()
  const prepared = dialog.getByRole('tabpanel', { name: 'Preprocessed CSV' })
  await expect(raw).toBeHidden()
  await expect(prepared.getByRole('list', { name: 'Preprocessed CSV required columns' }).locator('li')).toHaveText([
    'hour_of_day', 'day_of_week', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
    'type_PAYMENT', 'type_TRANSFER', 'log_amount', 'is_zero_amount',
    'is_merchant_origin', 'is_merchant_dest', 'isFraud',
  ])
  await expect(prepared).toContainText('cannot verify how the file was preprocessed')
  await expect(prepared.locator('details')).not.toHaveAttribute('open', '')
  await prepared.locator('summary').click()
  await expect(prepared.locator('details')).toContainText('All-zero type indicators mean unknown.')
  await expect(prepared.locator('details')).toContainText('Scaled values may fall outside [0, 1].')
  await expect(dialog).toContainText('500 MiB')
  for (const removed of ['Before you upload', 'Understanding your results', 'Additional Guidance',
    'Display Expectation', 'Original Inputs', 'optional', 'held-out']) {
    await expect(dialog).not.toContainText(removed)
  }
  const link = dialog.getByRole('link', { name: 'Download Raw CSV Example' })
  await expect(link).toHaveAttribute('href', '/api/v1/research/template')
  const downloading = page.waitForEvent('download')
  await link.click()
  expect((await downloading).suggestedFilename()).toBe('research_template.csv')
  await expect(page).toHaveURL(/researcher-upload$/)
})

test('tabs, expandable rules and closing preserve accessible focus behavior', async ({ page }) => {
  const dialog = await openGuide(page)
  const close = dialog.getByRole('button', { name: 'Close dataset instructions' })
  const link = dialog.getByRole('link', { name: 'Download Raw CSV Example' })
  const raw = dialog.getByRole('tab', { name: 'Raw CSV', exact: true })
  const prepared = dialog.getByRole('tab', { name: 'Preprocessed CSV' })
  await expect(close).toBeFocused()
  await expect(page.locator('#root')).toHaveAttribute('inert', '')
  await page.keyboard.press('Shift+Tab')
  await expect(link).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(close).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(dialog.getByRole('region')).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(raw).toBeFocused()
  for (const [key, tab] of [['ArrowRight', prepared], ['ArrowRight', raw], ['ArrowLeft', prepared], ['Home', raw], ['End', prepared]]) {
    await page.keyboard.press(key)
    await expect(tab).toBeFocused()
    await expect(tab).toHaveAttribute('aria-selected', 'true')
  }
  await page.keyboard.press('Tab')
  const panel = dialog.getByRole('tabpanel', { name: 'Preprocessed CSV' })
  await expect(panel.locator('summary')).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(panel.locator('details')).toHaveAttribute('open', '')
  await page.keyboard.press('Tab')
  await expect(link).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(panel.locator('summary')).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  await expect(page.locator('#root')).not.toHaveAttribute('inert', '')
  await expect(page.getByRole('button', { name: 'Upload Instructions' })).toBeFocused()
  await expect(page.locator('body')).not.toHaveCSS('overflow', 'hidden')
  await page.getByRole('button', { name: 'Upload Instructions' }).click()
  await expect(raw).toHaveAttribute('aria-selected', 'true')
  await close.click()
  await expect(dialog).toHaveCount(0)
  await page.getByRole('button', { name: 'Upload Instructions' }).click()
  await page.locator('.info-modal-overlay').click({ position: { x: 2, y: 2 } })
  await expect(dialog).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Upload Instructions' })).toBeFocused()
})

test('guide content fits desktop and mobile and scrolls to every section', async ({ page }, testInfo) => {
  const dialog = await openGuide(page)
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 844 })
    for (const format of ['Raw CSV', 'Preprocessed CSV']) {
      await dialog.getByRole('tab', { name: format, exact: true }).click()
      const panel = dialog.getByRole('tabpanel', { name: format, exact: true })
      const scroller = dialog.getByRole('region')
      await scroller.evaluate(element => { element.scrollTop = 0 })
      await expect(dialog.getByRole('tab', { name: format, exact: true })).toBeInViewport()
      expect(await scroller.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true)
      await page.screenshot({ path: testInfo.outputPath(`${width}-${format}-top.png`) })
      await panel.locator('summary').scrollIntoViewIfNeeded()
      if (!await panel.locator('details').evaluate(element => element.open)) await panel.locator('summary').click()
      await expect(panel.locator('details li').last()).toBeVisible()
      await dialog.getByRole('link', { name: 'Download Raw CSV Example' }).scrollIntoViewIfNeeded()
      await expect(dialog.getByRole('link', { name: 'Download Raw CSV Example' })).toBeInViewport()
      await page.screenshot({ path: testInfo.outputPath(`${width}-${format}-bottom.png`) })
      const box = await dialog.boundingBox()
      expect(box.x).toBeGreaterThanOrEqual(0)
      expect(box.x + box.width).toBeLessThanOrEqual(width)
    }
  }
})
