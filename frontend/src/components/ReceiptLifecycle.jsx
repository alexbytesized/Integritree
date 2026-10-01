import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { clearReceipt, connectPresence, disconnectPresence, resumePresence, suspendPresence } from '../receiptApi'

const detailRoutes = ['/results', '/user-results']
const flowRoutes = ['/upload', ...detailRoutes]

export default function ReceiptLifecycle({ children }) {
  const { pathname } = useLocation()
  const navigate = useNavigate()
  const previous = useRef(pathname)
  const [error, setError] = useState(null)
  useEffect(() => {
    const cleanup = event => setError(event.detail)
    const expired = () => { if (flowRoutes.includes(window.location.pathname)) navigate('/upload', { replace: true }) }
    const connect = () => { connectPresence().catch(() => {}) }
    window.addEventListener('receipt-cleanup', cleanup)
    window.addEventListener('receipt-session-expired', expired)
    window.addEventListener('receipt-session-change', connect)
    window.addEventListener('pagehide', suspendPresence)
    window.addEventListener('pageshow', resumePresence)
    connect()
    return () => {
      window.removeEventListener('receipt-cleanup', cleanup)
      window.removeEventListener('receipt-session-expired', expired)
      window.removeEventListener('receipt-session-change', connect)
      window.removeEventListener('pagehide', suspendPresence)
      window.removeEventListener('pageshow', resumePresence)
      disconnectPresence()
    }
  }, [navigate])
  useEffect(() => {
    const leftDetails = detailRoutes.includes(previous.current) && pathname === '/upload'
    if (!flowRoutes.includes(pathname) || leftDetails) clearReceipt().catch(() => {})
    previous.current = pathname
  }, [pathname])
  return <>
    {error && <div className="receipt-cleanup-error" role="alert">{error} <button onClick={() => clearReceipt().catch(() => {})}>Retry cleanup</button></div>}
    {children}
  </>
}
