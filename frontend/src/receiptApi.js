const base = '/api/v1/receipts'
const tokenKey = 'integritree-receipt-session'
export const receiptKey = 'integritree-receipt-id'
export const draftKey = (id, revision) => `integritree-receipt-draft:${id}:${revision}`
export const receiptToken = () => sessionStorage.getItem(tokenKey)
let socket, connectionPromise, opening, clearing, reconnectTimer
let generation = 0
let suspended = false
let cleanupError = null

function changed() { window.dispatchEvent(new Event('receipt-session-change')) }

export function forgetReceipt(token = receiptToken()) {
  if (token !== receiptToken()) return
  sessionStorage.removeItem(tokenKey)
  sessionStorage.removeItem(receiptKey)
  Object.keys(sessionStorage).filter(key => key.startsWith('integritree-receipt-draft:'))
    .forEach(key => sessionStorage.removeItem(key))
  disconnectPresence()
  changed()
}

export async function receiptApi(path = '', options = {}) {
  const token = receiptToken()
  const response = await fetch(path.startsWith('/api/') ? path : base + path, {
    ...options, headers: { 'X-Receipt-Session': token || '', ...options.headers },
  })
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    if (response.status === 410 && token === receiptToken()) {
      forgetReceipt(token)
      window.dispatchEvent(new Event('receipt-session-expired'))
    }
    const error = new Error(data.detail || `Request failed (${response.status})`)
    error.status = response.status
    error.issues = data.issues
    throw error
  }
  if (options.blob) return response.blob()
  return response.status === 204 ? null : response.json()
}

export function disconnectPresence() {
  clearTimeout(reconnectTimer)
  const previous = socket
  socket = null
  connectionPromise = null
  if (previous) previous.close()
}

export function connectPresence() {
  const token = receiptToken()
  if (!token || suspended) return Promise.resolve()
  if (connectionPromise) return connectionPromise
  const url = new URL(`${base}/sessions/presence`, window.location.href)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  const current = new WebSocket(url)
  socket = current
  connectionPromise = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { reject(new Error('Receipt connection unavailable. Please retry.')); current.close() }, 8000)
    current.onopen = () => current.send(JSON.stringify({ token }))
    current.onmessage = event => {
      if (JSON.parse(event.data).status === 'connected') { clearTimeout(timeout); resolve() }
    }
    current.onclose = event => {
      clearTimeout(timeout)
      reject(new Error('Receipt connection interrupted. Please retry.'))
      if (socket !== current) return
      socket = null
      connectionPromise = null
      if (event.code === 1008) {
        forgetReceipt(token)
        window.dispatchEvent(new Event('receipt-session-expired'))
      } else if (!suspended && token === receiptToken()) {
        reconnectTimer = setTimeout(() => { connectPresence().catch(() => {}) }, 1000)
      }
    }
  })
  return connectionPromise
}

export function suspendPresence() { suspended = true; disconnectPresence() }
export function resumePresence() { suspended = false; connectPresence().catch(() => {}) }

async function ensureSession() {
  if (!receiptToken()) {
    opening ||= receiptApi('/sessions', { method: 'POST' }).then(({ token }) => {
      sessionStorage.setItem(tokenKey, token)
      changed()
    }).finally(() => { opening = null })
    await opening
  }
  await connectPresence()
}

export async function uploadReceipt(file) {
  if (clearing) await clearing
  if (cleanupError) throw cleanupError
  const epoch = generation
  await ensureSession()
  if (epoch !== generation) throw new Error('Upload cancelled because you left the receipt flow.')
  const token = receiptToken()
  const data = await receiptApi(`?filename=${encodeURIComponent(file.name)}`, {
    method: 'POST', headers: { 'Content-Type': file.type }, body: file,
  })
  if (epoch !== generation || token !== receiptToken()) throw new Error('Receipt was cleared.')
  sessionStorage.setItem(receiptKey, data.id)
  changed()
  return data
}

export function clearReceipt() {
  if (clearing) return clearing
  generation++
  clearing = (async () => {
    if (opening) await opening
    const token = receiptToken()
    if (token) await receiptApi('/sessions/current', { method: 'DELETE' })
    forgetReceipt(token)
    cleanupError = null
    window.dispatchEvent(new CustomEvent('receipt-cleanup', { detail: null }))
  })().catch(error => {
    cleanupError = error
    window.dispatchEvent(new CustomEvent('receipt-cleanup', { detail: error.message }))
    throw error
  }).finally(() => { clearing = null })
  return clearing
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
