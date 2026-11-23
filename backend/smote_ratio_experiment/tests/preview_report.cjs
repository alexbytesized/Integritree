const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('../../../frontend/node_modules/playwright');

async function main() {
  if (!process.argv[2]) throw new Error('Usage: node preview_report.cjs <run-folder>');
  const run = path.resolve(process.argv[2]);
  const output = path.resolve(__dirname, '../runtime/qa', path.basename(run));
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const errors = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    await page.context().setOffline(true);
    page.on('pageerror', error => errors.push(String(error)));
    page.on('requestfailed', request => errors.push(request.url()));
    page.on('request', request => {
      if (/^https?:/.test(request.url())) errors.push(`Unexpected network request: ${request.url()}`);
    });
    await page.goto(pathToFileURL(path.join(run, 'report.html')).href);
    await page.getByRole('heading', { name: 'Ordinary SMOTE ratio comparison', exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'desktop.png') });
    for (const denominator of [200, 300, 500]) {
      await page.goto(pathToFileURL(path.join(run, `ratio_${denominator}`, 'report.html')).href);
      await page.getByRole('heading', { name: `Ordinary SMOTE ratio 1:${denominator}`, exact: true }).waitFor();
      const example = page.locator('#shap > details').first();
      await example.locator('summary').click();
      await example.scrollIntoViewIfNeeded();
      await page.waitForFunction(() => [...document.querySelectorAll('#shap details[open] img')]
        .every(img => img.complete && img.naturalWidth > 0));
      await example.screenshot({ path: path.join(output, `paired_${denominator}.png`) });
      await page.setViewportSize({ width: 390, height: 844 });
      if (await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)) {
        errors.push(`Mobile overflow for 1:${denominator}`);
      }
      await page.setViewportSize({ width: 1440, height: 1000 });
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(pathToFileURL(path.join(run, 'report.html')).href);
    await page.screenshot({ path: path.join(output, 'mobile.png') });
    if (await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)) errors.push('Summary mobile overflow');
    if (errors.length) throw new Error(errors.join('\n'));
    const result = { status: 'verified', offline: true, mobileOverflow: false, candidateReports: 3, screenshots: output };
    fs.writeFileSync(path.join(output, 'browser_verification.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
