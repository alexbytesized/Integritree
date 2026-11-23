// Offline report QA; browser profiles, screenshots, and temporary files stay local.
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('../../frontend/node_modules/playwright');

async function main() {
  if (!process.argv[2]) throw new Error('Usage: node preview_report.cjs <run-folder>');
  const run = path.resolve(process.argv[2]);
  const output = path.resolve(__dirname, '../runtime/qa', path.basename(run));
  const temporary = path.join(output, 'browser_temp');
  fs.mkdirSync(temporary, { recursive: true });
  process.env.TEMP = temporary;
  process.env.TMP = temporary;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const errors = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    page.on('pageerror', error => errors.push(String(error)));
    page.on('request', request => {
      if (/^https?:/.test(request.url())) errors.push('Unexpected network request');
    });
    await page.goto(pathToFileURL(path.join(run, 'report.html')).href);
    await page.getByRole('heading', { name: '200-tree regularization results', exact: true }).waitFor();
    for (const index of [0, 1]) {
      const table = page.locator('table').nth(index);
      if (await table.locator('tbody tr').count() !== 12) errors.push('Missing model rows');
      if (await table.locator('thead th').filter({ hasText: 'Pair average F1' }).count() !== 1) errors.push('Missing pair average F1');
    }
    await page.screenshot({ path: path.join(output, 'desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(output, 'mobile.png'), fullPage: true });
    if (await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)) errors.push('Mobile document overflow');
    if (errors.length) throw new Error(errors.join('\n'));
    const result = { status: 'verified', offline: true, desktopAndMobile: true, modelRowsPerPolicy: 12 };
    fs.writeFileSync(path.join(output, 'browser_verification.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
