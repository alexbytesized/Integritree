// Optional offline report QA with the project's existing Playwright and Chrome.
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('../../frontend/node_modules/playwright');

async function main() {
  if (!process.argv[2]) throw new Error('Usage: node preview_report.cjs <run-folder>');
  const run = path.resolve(process.argv[2]);
  const output = path.resolve(__dirname, '../runtime/qa', path.basename(run));
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const errors = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.on('pageerror', e => errors.push(String(e)));
    page.on('requestfailed', r => errors.push(r.url()));
    page.on('request', r => { if (/^https?:/.test(r.url())) errors.push(r.url()); });
    await page.goto(pathToFileURL(path.join(run, 'report.html')).href);
    await page.getByRole('heading', { name: 'Experimental three-stage validation', exact: true }).waitFor();
    const brokenImages = await page.locator('img').evaluateAll(images => images.some(i => !i.complete || i.naturalWidth === 0));
    if (brokenImages) errors.push('Unloaded chart image');
    await page.screenshot({ path: path.join(output, 'desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(output, 'mobile.png'), fullPage: true });
    if (await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)) errors.push('Mobile page overflow');
    if (errors.length) throw new Error(errors.join('\n'));
    const result = { status: 'verified', offline: true, mobileOverflow: false, screenshots: output };
    fs.writeFileSync(path.join(output, 'browser_verification.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
