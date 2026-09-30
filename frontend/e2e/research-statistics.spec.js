import { test, expect } from '@playwright/test'

async function results(page, statisticalTest) {
  const metrics = Object.fromEntries(['precision', 'recall', 'f1', 'mcc', 'pr_auc'].map(key => [key, { value: .8 }]))
  const model = { metrics, actual_fraud: 20, confusion_matrix: { tp: 15, tn: 75, fp: 5, fn: 5 } }
  const job = { status: 'complete', filename: 'statistics.csv', rows_processed: 100, threshold: .43,
    evaluation: { models: { rf: model, rf_smote: model }, descriptive_comparisons: metrics, statistical_test: statisticalTest } }
  await page.route('**/api/v1/research/analyses/statistics**', route => route.fulfill({ json:
    route.request().url().includes('/records') ? { records: [], total: 0, page: 1 } : job }))
  await page.goto('/researcher?analysis=statistics')
  await expect(page.getByRole('heading', { name: 'Results Overview' })).toBeVisible()
}

const table = { both_correct: 76, rf_only_correct: 5, smote_only_correct: 5, both_incorrect: 14 }
const smallTest = { table, discordant_pairs: 10, statistic: .1, p_value: .751829634, approximation_caution: true,
  supplementary: { method: 'exact_binomial_two_sided', p_value: 1 } }

async function expectAlignedCards(page) {
  const cards = await page.locator('.mcnemar-table .table-value').evaluateAll(elements => elements.map(element => {
    const box = element.getBoundingClientRect()
    return { height: box.height, width: box.width, bottom: box.bottom }
  }))
  const interpretation = await page.locator('.mcnemar-result .interpretation').boundingBox()
  for (const card of cards) {
    expect(Math.abs(card.height - cards[0].height)).toBeLessThan(1)
    expect(Math.abs(card.width - cards[0].width)).toBeLessThan(1)
  }
  for (const card of cards.slice(2)) expect(Math.abs(card.bottom - interpretation.y - interpretation.height)).toBeLessThan(1)
}

test('metric order and P-Value modal show current values with accessible desktop and mobile controls', async ({ page }, testInfo) => {
  await results(page, smallTest)
  await expect(page.locator('.difference-label')).toHaveText(['Precision', 'Recall', 'F1-Score', 'MCC', 'PR-AUC'])
  await expect(page.locator('.interpretation')).toHaveText('P-value >= 0.05: Fail to reject the null hypothesis. The difference between the benchmark RF model and the RF-SMOTE model is not statistically significant.')
  await expect(page.locator('.interpretation p')).toHaveCount(1)
  await expect(page.locator('.interpretation p')).toHaveCSS('text-align', 'justify')
  for (const width of [1920, 1024, 1440]) {
    await page.setViewportSize({ width, height: 1000 })
    await expectAlignedCards(page)
  }
  const open = page.getByRole('button', { name: 'View statistical test details', exact: true })
  await page.locator('.mcnemar-statistics-container').scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('statistics-desktop.png') })
  const iconBox = await open.boundingBox()
  const valueBox = await page.locator('.pvalue-box').boundingBox()
  expect(iconBox.x).toBeGreaterThan(valueBox.x + valueBox.width / 2)
  expect(Math.abs(valueBox.x + valueBox.width - iconBox.x - iconBox.width)).toBeLessThan(1)
  const labelBox = await page.locator('.pvalue-label span').boundingBox()
  expect(iconBox.y + iconBox.height / 2).toBeCloseTo(labelBox.y + labelBox.height / 2, 1)
  await expect(open).toHaveAttribute('title', 'View statistical test details')
  await expect(page.locator('.pvalue-box .info-button')).toHaveCount(0)
  await open.focus()
  await open.press('Enter')
  const dialog = page.getByRole('dialog', { name: 'Statistical Test Details', exact: true })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole('heading', { level: 3 })).toHaveText(['Discordant Pairs',
    'Chi-Square Distribution', 'Primary P-Value',
    'Exact Binomial Distribution', 'Supplementary P-Value'])
  await expect(dialog).toContainText('A supplementary exact binomial test is provided when there are 1–24 discordant pairs because the chi-square approximation may be less reliable with such small counts.')
  await expect(dialog).toContainText('Total: 10')
  await expect(dialog).toContainText('Benchmark RF correct only (B): 5')
  await expect(dialog).toContainText('RF-SMOTE correct only (C): 5')
  for (const text of ['Chi-square statistic: 0.1', 'Primary p-value: 7.518296e-1',
    'Trials: 10', 'Probability: 0.5', 'Supplementary p-value: 1.00000']) {
    const item = dialog.getByRole('listitem').filter({ hasText: text })
    await expect(item).toHaveText(text)
    await expect(item.locator('strong')).toHaveCSS('font-weight', '700')
  }
  for (const name of ['Primary P-Value', 'Supplementary P-Value']) {
    const section = dialog.locator('section').filter({ has: page.getByRole('heading', { name, exact: true }) })
    await expect(section.locator('p')).toHaveCount(0)
  }
  const close = dialog.getByRole('button', { name: 'Close Statistical Test Details information' })
  await expect(close).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(dialog.getByRole('region', { name: 'Statistical test details' })).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(close).toBeFocused()
  await page.screenshot({ path: testInfo.outputPath('pvalue-desktop.png') })
  await page.keyboard.press('Escape')
  await expect(open).toBeFocused()
  await expect(dialog).toHaveCount(0)
  await expect(page.locator('body')).not.toHaveCSS('overflow', 'hidden')
  await open.click()
  await page.locator('.info-modal-overlay').click({ position: { x: 2, y: 2 } })
  await expect(dialog).toHaveCount(0)
  await page.setViewportSize({ width: 390, height: 844 })
  await open.click()
  const box = await dialog.boundingBox()
  expect(box.x).toBeGreaterThanOrEqual(0)
  expect(box.x + box.width).toBeLessThanOrEqual(390)
  await page.screenshot({ path: testInfo.outputPath('pvalue-mobile.png') })
  await dialog.getByRole('listitem').filter({ hasText: 'Supplementary p-value:' }).scrollIntoViewIfNeeded()
  await expect(dialog.getByText('Supplementary p-value:', { exact: false })).toBeInViewport()
  await page.screenshot({ path: testInfo.outputPath('pvalue-mobile-footer.png') })
  await close.click()
  await expect(open).toBeFocused()
  await page.screenshot({ path: testInfo.outputPath('statistics-mobile.png') })
})

for (const [description, pValue] of [['significant', .04999999], ['boundary', .05], ['not significant', .5838824]]) {
  test(`primary ${description} result uses the unrounded p-value and omits unavailable exact results`, async ({ page }) => {
    await results(page, { table: { ...table, rf_only_correct: 13, smote_only_correct: 17 },
      discordant_pairs: 30, statistic: .3, p_value: pValue, supplementary: null, approximation_caution: false })
    await expect(page.locator('.interpretation')).toHaveText(pValue < .05
      ? 'P-value < 0.05: Reject the null hypothesis. The difference between the benchmark RF model and the RF-SMOTE model is statistically significant.'
      : 'P-value >= 0.05: Fail to reject the null hypothesis. The difference between the benchmark RF model and the RF-SMOTE model is not statistically significant.')
    await expectAlignedCards(page)
    await page.getByRole('button', { name: 'View statistical test details', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toContainText('Total: 30')
    await expect(dialog).toContainText('Chi-square statistic: 0.3')
    await expect(dialog.getByRole('heading', { name: 'Exact Binomial Distribution', exact: true })).toHaveCount(0)
    await expect(dialog.getByRole('heading', { name: 'Supplementary P-Value', exact: true })).toHaveCount(0)
  })
}

test('zero discordant pairs preserves the p-value convention without inventing a statistic', async ({ page }) => {
  await results(page, { table: { ...table, rf_only_correct: 0, smote_only_correct: 0 }, discordant_pairs: 0,
    statistic: null, p_value: 1, supplementary: null })
  await page.getByRole('button', { name: 'View statistical test details', exact: true }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog).toContainText('Total: 0')
  await expect(dialog).toContainText('Chi-square statistic: Not available')
  await expect(dialog.getByRole('listitem').filter({ hasText: 'Primary p-value:' })).toHaveText('Primary p-value: 1.000000e+0')
  await expect(dialog).not.toContainText('NaN')
  await expect(dialog.getByRole('heading', { name: 'Exact Binomial Distribution', exact: true })).toHaveCount(0)
})
