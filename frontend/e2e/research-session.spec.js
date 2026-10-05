import { test, expect } from '@playwright/test'

const tokenKey = 'integritree-research-session'
const analysisKey = 'integritree-research-analysis'
const csv = 'step,type,amount,nameOrig,nameDest,isFraud\n1,TRANSFER,100,C_SYNTHETIC,C_DEST,0\n'

async function setup(page, responses, token = 'expired-token') {
  const state = { uploads: [], sessions: [], responses: [...responses] }
  await page.route('**/api/v1/research/**', async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (path.endsWith('/sessions')) {
      state.sessions.push(await page.evaluate(({ tokenKey, analysisKey }) => ({
        token: sessionStorage.getItem(tokenKey), analysis: sessionStorage.getItem(analysisKey),
      }), { tokenKey, analysisKey }))
      return route.fulfill({ status: 201, json: { token: `fresh-token-${state.sessions.length}` } })
    }
    if (path.endsWith('/analyses')) {
      state.uploads.push({ token: request.headers()['x-research-session'], body: request.postData() })
      const status = state.responses.shift() ?? 202
      if (status === 'network') return route.abort('failed')
      if (status !== 202) return route.fulfill({ status, json: { detail: status === 410
        ? 'Session expired. Upload the CSV again to start a new analysis.' : `Synthetic rejection ${status}` } })
      return route.fulfill({ status: 202, json: { id: 'recovered', status: 'queued' } })
    }
    return route.fulfill({ json: { id: 'recovered', status: 'queued', rows_processed: 0 } })
  })
  await page.goto('/researcher-upload')
  await page.evaluate(({ tokenKey, analysisKey, token }) => {
    if (token) {
      sessionStorage.setItem(tokenKey, token)
      sessionStorage.setItem(analysisKey, 'old-analysis')
    }
  }, { tokenKey, analysisKey, token })
  await page.locator('input[type=file]').setInputFiles({ name: 'synthetic.csv', mimeType: 'text/csv', buffer: Buffer.from(csv) })
  return state
}

test('upload renews a stale session once and resends the selected CSV', async ({ page }) => {
  const state = await setup(page, [410, 202])
  await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
  await expect(page).toHaveURL(/researcher\?analysis=recovered$/)
  expect(state.uploads).toEqual([
    { token: 'expired-token', body: csv }, { token: 'fresh-token-1', body: csv },
  ])
  expect(state.sessions).toEqual([{ token: null, analysis: null }])
  expect(await page.evaluate(key => sessionStorage.getItem(key), analysisKey)).toBe('recovered')
  expect(await page.evaluate(key => sessionStorage.getItem(key), tokenKey)).toBe('fresh-token-1')
})

test('repeated expiry stops after one retry and permits a manual retry', async ({ page }) => {
  const state = await setup(page, [410, 410, 202])
  await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
  await expect(page.getByRole('alert')).toContainText('Session expired')
  await expect(page.getByRole('button', { name: 'ANALYZE FILE' })).toBeEnabled()
  await expect(page.getByText('synthetic.csv', { exact: true })).toBeVisible()
  expect(state.uploads).toHaveLength(2)
  expect(state.sessions).toHaveLength(1)
  expect(await page.evaluate(({ tokenKey, analysisKey }) => [sessionStorage.getItem(tokenKey), sessionStorage.getItem(analysisKey)], { tokenKey, analysisKey })).toEqual([null, null])
  await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
  await expect(page).toHaveURL(/researcher\?analysis=recovered$/)
  expect(state.uploads).toHaveLength(3)
  expect(state.sessions).toHaveLength(2)
})

for (const rejection of [400, 413, 429, 'network']) {
  test(`upload does not automatically retry ${rejection}`, async ({ page }) => {
    const state = await setup(page, [rejection], 'valid-token')
    await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
    await expect(page.getByRole('alert')).toContainText(rejection === 'network' ? 'Upload connection failed' : `Synthetic rejection ${rejection}`)
    await expect(page.getByRole('button', { name: 'ANALYZE FILE' })).toBeEnabled()
    expect(state.uploads).toHaveLength(1)
    expect(state.sessions).toHaveLength(0)
    expect(await page.evaluate(key => sessionStorage.getItem(key), tokenKey)).toBe('valid-token')
  })
}

for (const token of [null, 'valid-token']) {
  test(`successful upload with ${token ? 'existing' : 'new'} session sends CSV once`, async ({ page }) => {
    const state = await setup(page, [202], token)
    await page.getByRole('button', { name: 'ANALYZE FILE' }).click()
    await expect(page).toHaveURL(/researcher\?analysis=recovered$/)
    expect(state.uploads).toEqual([{ token: token || 'fresh-token-1', body: csv }])
    expect(state.sessions).toHaveLength(token ? 0 : 1)
  })
}
