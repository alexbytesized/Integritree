import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import TransactionDetails from '../components/TransactionDetails'
import ResearchAnalysisLoading from '../components/ResearchAnalysisLoading'
import ReceiptParticles from '../components/ReceiptParticles'
import ScreenshotPreview from '../components/ScreenshotPreview'
import ReturnButton from '../components/ReturnButton'
import UploadErrorPage from '../components/UploadErrorPage'
import stars from '../assets/stars.png'
import { receiptApi, receiptKey, clearReceipt, draftKey } from '../receiptApi'
import { useReceipt, useReceiptImage } from '../useReceipt'
import './UploadDetails.css'
import './ResearcherResultsPage.css'
import './ReceiptFlow.css'

import { WORKFLOWS, TRANSACTION_TYPES, minuteTime } from '../receiptDisplay'

function ConfirmationForm({ job, onSubmitted, onError, onClear, clearing }) {
  const live = useRef(true)
  useEffect(() => { live.current = true; return () => { live.current = false } }, [])
  const key = draftKey(job.id, job.revision)
  const [form, setForm] = useState(() => {
    const seed = job.confirmed_fields || job.candidates
    const category = job.confirmed_fields?.category || WORKFLOWS[job.workflow].category
    const defaults = TRANSACTION_TYPES[category]
    const value = { amount: seed.amount ?? '', date: seed.date ?? '', time: minuteTime(seed.time),
      reference: seed.reference ?? '', category,
      origin_role: job.confirmed_fields?.origin_role || defaults.origin_role,
      destination_role: job.confirmed_fields?.destination_role || defaults.destination_role }
    try {
      const draft = JSON.parse(sessionStorage.getItem(key))
      if (draft && Object.keys(value).every(name => typeof draft[name] === 'string') &&
          TRANSACTION_TYPES[draft.category] && ['client', 'merchant'].includes(draft.origin_role) &&
          ['client', 'merchant'].includes(draft.destination_role)) return { ...draft, time: minuteTime(draft.time) }
    } catch { /* Ignore a damaged draft and use extracted/confirmed values. */ }
    return value
  })
  const [sending, setSending] = useState(false)
  const change = (name, value) => {
    const next = { ...form, [name]: value }
    if (name === 'category') {
      next.origin_role = TRANSACTION_TYPES[value].origin_role
      next.destination_role = TRANSACTION_TYPES[value].destination_role
    }
    sessionStorage.setItem(key, JSON.stringify(next))
    setForm(next)
  }
  const submit = async event => {
    event.preventDefault()
    setSending(true)
    try {
      const value = await receiptApi(`/${job.id}/confirm`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
        expected_revision: job.revision, confirmed: true,
        fields: { ...form, time: minuteTime(form.time), reference: form.reference || null, workflow: job.workflow,
          currency: 'PHP', timezone: 'Asia/Manila' },
      }) })
      sessionStorage.removeItem(key)
      if (live.current) onSubmitted(value)
    } catch (error) {
      onError([error.message, ...(error.issues || []).map(issue => `${issue.loc.join('.') || 'Receipt'}: ${issue.msg}`)].join(' '))
    } finally { setSending(false) }
  }
  return <form className="receipt-confirmation" onSubmit={submit}>
    <TransactionDetails value={form} onChange={change} disabled={sending} />
    <div className="proceed-button-container receipt-form-actions">
      <button className="proceed-button" disabled={sending || clearing}>{sending ? 'Submitting...' : 'Proceed'}</button>
      <button type="button" className="proceed-button receipt-button-outline" disabled={clearing} onClick={onClear}>Clear Receipt</button>
    </div>
  </form>
}

export default function UploadDetails() {
  const [params] = useSearchParams()
  const id = params.get('receipt') || sessionStorage.getItem(receiptKey)
  return <ReceiptDetails key={id || 'empty'} id={id} />
}

function ReceiptDetails({ id }) {
  const navigate = useNavigate()
  const [refresh, setRefresh] = useState(0)
  const { job, error: loadError } = useReceipt(id, refresh)
  const image = useReceiptImage(id, !!job?.workflow)
  const [error, setError] = useState('')
  const [working, setWorking] = useState(false)
  const pending = useRef(false)
  const action = async fn => {
    if (pending.current) return
    pending.current = true
    setWorking(true); setError('')
    try { await fn() } catch (err) { setError(err.message) }
    finally { pending.current = false; setWorking(false) }
  }
  const clear = () => action(async () => { await clearReceipt(); navigate('/upload') })
  const retry = () => action(async () => { await receiptApi(`/${id}/retry`, { method: 'POST' }); setRefresh(v => v + 1) })
  const ready = job && !job.busy && job.workflow && ['awaiting_confirmation', 'complete', 'failed'].includes(job.status)
  if (!loadError && ((id && !job) || job?.busy)) return <ResearchAnalysisLoading
    title="Receipt analysis" message={job?.busy
      ? (['uploading', 'queued', 'extracting'].includes(job.status) ? 'Reading your receipt...' : 'Preparing your predictions and explanations...')
      : 'Loading your receipt...'} onReturn={clear} returning={working} error={error} />
  if (job?.status === 'unsupported' || (job?.status === 'failed' && !job.workflow)) return <UploadErrorPage
    message={job.error_code === 'unsupported_receipt_layout'
      ? `The receipt layout of the uploaded photo (${job.filename || 'unnamed photo'}) is unclear or unsupported. Use a complete GCash Express Send, Pay Online, or bank-account transfer screenshot.`
      : job.error || 'The receipt could not be processed. Please try another screenshot.'}
    clearLabel="Clear Receipt" onClear={clear} onReturn={clear} busy={working} actionError={error || loadError}
    onRetry={job.status === 'failed' ? retry : undefined} />
  return <div className="researcher-results-wrapper">
    <ReceiptParticles />
    <ReturnButton onClick={clear} />
    <main className="user-details-page receipt-page">
      <div className="results-overview-title"><div className="section-heading"><h1>Confirm Details</h1><img src={stars} alt="" /></div></div>
      {(error || loadError) && <p role="alert">{error || loadError}</p>}
      {!id && <p>Upload a receipt to begin. <Link to="/upload">Choose screenshot</Link></p>}
      {job && <>
        <ScreenshotPreview fileName={job.filename} imageUrl={image} />
        {job.error && <p role="alert">{job.error}</p>}
        {(!ready || job.status === 'failed') && <div className="receipt-actions">
          {!ready && <button className="proceed-button receipt-button-outline" disabled={working} onClick={clear}>Clear Receipt</button>}
          {job.status === 'failed' && !job.busy && <button disabled={working} onClick={retry}>Retry processing</button>}
        </div>}
        {ready && <ConfirmationForm key={`${id}-${job.revision}`} job={job} onError={setError} onClear={clear} clearing={working}
          onSubmitted={() => navigate(`/user-results?receipt=${id}`)} />}
      </>}
    </main>
  </div>
}
