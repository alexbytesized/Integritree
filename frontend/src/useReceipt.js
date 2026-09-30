import { useEffect, useState } from 'react'
import { receiptApi } from './receiptApi'

export function useReceipt(id, refresh = 0) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let live = true
    let timer
    const poll = async () => {
      try {
        const value = await receiptApi(`/${id}`)
        if (!live) return
        setError('')
        setJob(value)
        if (value.busy) timer = setTimeout(poll, 900)
      } catch (err) { if (live) setError(err.message) }
    }
    if (id) poll()
    return () => { live = false; clearTimeout(timer) }
  }, [id, refresh])
  return { job, error }
}

export function useReceiptImage(id, ready) {
  const [image, setImage] = useState(null)
  useEffect(() => {
    let live = true
    let url
    if (id && ready) receiptApi(`/${id}/image`, { blob: true }).then(blob => {
      url = URL.createObjectURL(blob)
      if (live) setImage(url)
      else URL.revokeObjectURL(url)
    }).catch(() => {})
    return () => { live = false; if (url) URL.revokeObjectURL(url) }
  }, [id, ready])
  return image
}
