import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import ScreenshotPreview from '../components/ScreenshotPreview'
import ReturnButton from '../components/ReturnButton'
import { receiptApi, receiptKey, clearReceipt } from '../receiptApi'
import { useReceipt, useReceiptImage } from '../useReceipt'
import './UploadDetails.css'
import './ReceiptFlow.css'

import { WORKFLOWS } from '../receiptDisplay'

function ConfirmationForm({ job, onSubmitted, onError }) {
  const seed = job.confirmed_fields || job.candidates
  const [form, setForm] = useState({ amount: seed.amount || '', date: seed.date || '', time: seed.time || '', reference: seed.reference || '' })
  const [roles, setRoles] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const [sending, setSending] = useState(false)
  const workflow = WORKFLOWS[job.workflow]
  const submit = async event => {
    event.preventDefault()
    setSending(true)
    try {
      const value = await receiptApi(`/${job.id}/confirm`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
        expected_revision: job.revision, confirmed,
        fields: { ...form, reference: form.reference || null, workflow: job.workflow, category: workflow.category,
          origin_role: 'personal_wallet', destination_role: workflow.destination, wallet_funded: roles,
          currency: 'PHP', timezone: 'Asia/Manila' },
      }) })
      onSubmitted(value)
    } catch (error) {
      onError([error.message, ...(error.issues || []).map(issue => `${issue.loc.join('.') || 'Receipt'}: ${issue.msg}`)].join(' '))
    } finally { setSending(false) }
  }
  return <form className="receipt-form td-card" onSubmit={submit}>
    <div className="td-card-header">Confirm transaction details</div>
    <div className="td-card-body">
      <p><strong>{workflow.label}</strong> maps to {workflow.category}</p>
      <p>Check every value against your screenshot. Complete any unreadable fields. The amount must exclude fees.</p>
      <label>Principal amount (PHP)<input name="amount" inputMode="decimal" required maxLength={32} pattern="[0-9]+([.][0-9]{1,2})?" value={form.amount} onChange={e => { setForm({ ...form, amount: e.target.value }); setConfirmed(false) }} /></label>
      <label>Transaction date (Manila)<input name="date" type="date" required value={form.date} onChange={e => { setForm({ ...form, date: e.target.value }); setConfirmed(false) }} /></label>
      <label>Transaction time (Manila)<input name="time" type="time" step="1" required value={form.time} onChange={e => { setForm({ ...form, time: e.target.value }); setConfirmed(false) }} /></label>
      <label>Reference (optional)<input name="reference" maxLength={256} value={form.reference} onChange={e => { setForm({ ...form, reference: e.target.value }); setConfirmed(false) }} /></label>
      <label className="receipt-check"><input type="checkbox" required checked={roles} onChange={e => { setRoles(e.target.checked); setConfirmed(false) }} />I confirm this completed transaction was funded from my personal GCash wallet and the destination is a {workflow.role.toLowerCase()}.</label>
      <label className="receipt-check"><input type="checkbox" required checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />I have reviewed and confirm these transaction details.</label>
      <p>Experimental predictions use PaySim-trained models. They do not verify receipt authenticity or establish real-world fraud.</p>
      <button className="proceed-button" disabled={sending || !roles || !confirmed}>{sending ? 'Submitting...' : 'Confirm and analyze'}</button>
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
  const action = async fn => { setError(''); try { await fn() } catch (err) { setError(err.message) } }
  const ready = job && !job.busy && job.workflow && ['awaiting_confirmation', 'complete', 'failed'].includes(job.status)
  return <div className="researcher-results-wrapper">
    <ReturnButton to="/upload" />
    <main className="user-details-page receipt-page">
      <h1>Review your receipt</h1>
      {(error || loadError) && <p role="alert">{error || loadError}</p>}
      {!id && <p>Upload a receipt to begin. <Link to="/upload">Choose screenshot</Link></p>}
      {id && !job && !loadError && <p role="status">Loading receipt...</p>}
      {job && <>
        <ScreenshotPreview fileName={job.filename} imageUrl={image} />
        {job.busy && <p role="status">{['queued', 'extracting', 'uploading'].includes(job.status) ? 'Reading the receipt locally...' : 'Preparing predictions and explanations...'}</p>}
        {job.error && <p role="alert">{job.error}</p>}
        <div className="receipt-actions">
          <button disabled={job.busy} onClick={() => action(async () => { await clearReceipt(id); navigate('/upload') })}>Clear receipt</button>
          {job.status === 'failed' && !job.busy && <button onClick={() => action(async () => { await receiptApi(`/${id}/retry`, { method: 'POST' }); setRefresh(v => v + 1) })}>Retry processing</button>}
          {job.status === 'complete' && !job.busy && <Link to={`/user-results?receipt=${id}`}>View results</Link>}
        </div>
        {ready && <ConfirmationForm key={`${id}-${job.revision}`} job={job} onError={setError}
          onSubmitted={() => navigate(`/user-results?receipt=${id}`)} />}
        <p className="receipt-retention">Temporary storage: Clear removes this receipt and its results. Backend shutdown/restart also clears them. Download results first.</p>
      </>}
    </main>
  </div>
}
