const base = '/api/v1/research'
const tokenKey = 'integritree-research-session'
export const analysisKey = 'integritree-research-analysis'

export async function api(path, options = {}) {
  const response = await fetch(path.startsWith('/api/') ? path : base + path, {
    ...options, headers: { 'X-Research-Session': sessionStorage.getItem(tokenKey) || '', ...options.headers },
  })
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    if (response.status === 410) {
      sessionStorage.removeItem(tokenKey)
      sessionStorage.removeItem(analysisKey)
      if (window.location.pathname !== '/researcher-upload') window.location.replace('/researcher-upload?expired=1')
    }
    const error = new Error(data.detail || `Request failed (${response.status})`)
    error.status = response.status
    throw error
  }
  if (options.blob) return response.blob()
  return response.status === 204 ? null : response.json()
}

export async function uploadCsv(file, onProgress) {
  if (!sessionStorage.getItem(tokenKey)) {
    const { token } = await api('/sessions', { method: 'POST' })
    sessionStorage.setItem(tokenKey, token)
  }
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${base}/analyses?filename=${encodeURIComponent(file.name)}`)
    xhr.setRequestHeader('Content-Type', 'text/csv')
    xhr.setRequestHeader('X-Research-Session', sessionStorage.getItem(tokenKey))
    xhr.upload.onprogress = (event) => onProgress(event.lengthComputable ? Math.round(event.loaded / event.total * 100) : null)
    xhr.onerror = () => reject(new Error('Upload connection failed. Please retry.'))
    xhr.onload = () => {
      let data
      try { data = JSON.parse(xhr.responseText) } catch { reject(new Error('Invalid server response.')); return }
      if (xhr.status >= 200 && xhr.status < 300) {
        sessionStorage.setItem(analysisKey, data.id)
        resolve(data)
      } else {
        if (xhr.status === 410) sessionStorage.removeItem(tokenKey)
        reject(new Error(data.detail || 'Upload failed. Please retry.'))
      }
    }
    xhr.send(file)
  })
}

export async function downloadAnalysis(id) {
  await api(`/analyses/${id}/exports`, { method: 'POST' })
  for (;;) {
    const job = await api(`/analyses/${id}`)
    if (job.export.status === 'failed') throw new Error(job.export.error)
    if (job.export.status === 'complete') break
    await new Promise((resolve) => setTimeout(resolve, 1000))
  }
  const blob = await api(`/analyses/${id}/exports/download`, { blob: true })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = 'integritree_results.zip'
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 30000)
}

export const label = (value) => value === 1 ? 'Fraudulent' : 'Legitimate'
export const metric = (item, coefficient = false) => item?.value == null ? `N/A (${item?.reason || 'unavailable'})`
  : coefficient ? item.value.toFixed(4) : `${(item.value * 100).toFixed(2)}%`
export const outcomeName = (pred, actual) => pred === 1 ? (actual === 1 ? 'True Positive' : 'False Positive')
  : (actual === 0 ? 'True Negative' : 'False Negative')
