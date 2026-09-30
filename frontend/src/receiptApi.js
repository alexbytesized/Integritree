const base = '/api/v1/receipts'
const tokenKey = 'integritree-receipt-session'
export const receiptKey = 'integritree-receipt-id'

export async function receiptApi(path = '', options = {}) {
  const response = await fetch(path.startsWith('/api/') ? path : base + path, {
    ...options, headers: { 'X-Receipt-Session': sessionStorage.getItem(tokenKey) || '', ...options.headers },
  })
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    if (response.status === 410) {
      sessionStorage.removeItem(tokenKey)
      sessionStorage.removeItem(receiptKey)
    }
    const error = new Error(data.detail || `Request failed (${response.status})`)
    error.status = response.status
    error.issues = data.issues
    throw error
  }
  if (options.blob) return response.blob()
  return response.status === 204 ? null : response.json()
}

export async function uploadReceipt(file) {
  if (!sessionStorage.getItem(tokenKey)) {
    const { token } = await receiptApi('/sessions', { method: 'POST' })
    sessionStorage.setItem(tokenKey, token)
  }
  const data = await receiptApi(`?filename=${encodeURIComponent(file.name)}`, {
    method: 'POST', headers: { 'Content-Type': file.type }, body: file,
  })
  sessionStorage.setItem(receiptKey, data.id)
  return data
}

export async function clearReceipt(id) {
  try { await receiptApi(`/${id}`, { method: 'DELETE' }) }
  catch (error) { if (![404, 410].includes(error.status)) throw error }
  sessionStorage.removeItem(receiptKey)
}

export async function downloadReceipt(id, revision) {
  const blob = await receiptApi(`/${id}/download?revision=${revision}`, { blob: true })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = 'integritree_receipt.zip'
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 30000)
}
