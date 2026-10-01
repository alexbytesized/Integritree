import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import stars from '../assets/stars.png'
import ResearchAnalysisLoading from '../components/ResearchAnalysisLoading'
import TransactionAmountTooltip from '../components/TransactionAmountTooltip'
import ReturnButton from '../components/ReturnButton'
import ScreenshotPreview from '../components/ScreenshotPreview'
import TabBar from '../components/TabBar'
import ModelResultView from '../components/ModelResultView'
import BothModelsView from '../components/BothModelsView'
import ShapModal from '../components/ShapModal'
import { MODEL_NAMES, booleanText, weekdayText, shapSummary } from '../researchDisplay'
import { receiptApi, receiptKey, clearReceipt, downloadReceipt } from '../receiptApi'
import { useReceipt, useReceiptImage } from '../useReceipt'
import { TRANSACTION_TYPES, roleLabel, receiptDate, receiptTime } from '../receiptDisplay'
import './TransactionDetailsPage.css'
import './UploadDetails.css'
import './ResearcherResultsPage.css'
import './UserResultsPage.css'
import './ReceiptFlow.css'

const TABS = [{ key: 'details', label: 'Transaction Details' }, { key: 'rfSmote', label: 'RF-SMOTE' },
  { key: 'benchmark', label: 'Benchmark RF' }, { key: 'both', label: 'Both Models' }]
const Field = ({ label, value, amountHelp = false }) => <div className="td-field-row"><span className="td-field-label">{label}</span><div className={`td-field-value${amountHelp ? ' receipt-amount-control' : ''}`}>{value == null || value === '' ? 'Not supplied' : String(value)}{amountHelp && <TransactionAmountTooltip />}</div></div>

export default function UserResultsPage() {
  const [params] = useSearchParams()
  const id = params.get('receipt') || sessionStorage.getItem(receiptKey)
  return <ReceiptResults key={id || 'empty'} id={id} />
}

function ReceiptResults({ id }) {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('details')
  const [shapModel, setShapModel] = useState(null)
  const [refresh, setRefresh] = useState(0)
  const [error, setError] = useState('')
  const [working, setWorking] = useState(false)
  const { job, error: loadError } = useReceipt(id, refresh)
  const image = useReceiptImage(id, !!job?.workflow)
  const record = job?.result
  const action = async fn => { setWorking(true); setError(''); try { await fn() } catch (err) { setError(err.message) } finally { setWorking(false) } }
  const retry = () => action(async () => { await receiptApi(`/${id}/retry`, { method: 'POST' }); setRefresh(v => v + 1) })
  if (!loadError && ((id && !job) || job?.busy)) return <ResearchAnalysisLoading title="Receipt results"
    message="Preparing your predictions and explanations..." returnTo={`/results?receipt=${id}`} returnLabel="Return to receipt" />
  if (!record || job.status !== 'complete') return <main className="user-details-page receipt-page">
    <h1>Receipt results</h1><p role={loadError || job?.error ? 'alert' : 'status'}>{loadError || job?.error || (job?.busy ? 'Preparing your predictions and explanations...' : 'No confirmed receipt results yet.')}</p>
    {error && <p role="alert">{error}</p>}
    {job?.status === 'failed' && <button disabled={working || job.busy} onClick={retry}>Retry processing</button>}
    <p><Link to={id ? `/results?receipt=${id}` : '/upload'}>{id ? 'Return to receipt' : 'Upload a receipt'}</Link></p>
  </main>
  const modelProps = name => ({ modelName: MODEL_NAMES[name], prediction: record[name].predicted_label === 1 ? 'Fraudulent' : 'Legitimate',
    riskScore: record[name].score * 100, threshold: record.threshold * 100,
    shapSummary: shapSummary(record.explanation, name, record.derived, 'receipt'), onOpenShap: () => setShapModel(name) })
  const original = record.original
  const derived = record.derived
  return <div className="researcher-results-wrapper">
    <ReturnButton to={`/results?receipt=${id}`} />
    <main className="researcher-results-page receipt-results-page">
      <section className="results-overview-section">
        <div className="results-overview-title"><div className="section-heading"><h1>Results Overview</h1><img src={stars} alt="" /></div></div>
        {loadError && <p role="alert">{loadError}</p>}
        <ScreenshotPreview fileName={job.filename} imageUrl={image} />
        <div className="user-results-tabs-wrapper"><TabBar tabs={TABS} active={activeTab} onChange={tab => { setShapModel(null); setActiveTab(tab) }} /></div>
      </section>
      <section className="td-content-section">
          <div className="td-record-heading"><h2 className="td-record-title">Transaction Record</h2><p className="td-record-id"><span className="td-record-id-label">Transaction ID:</span> {original.reference?.trim() || 'Not provided'}</p></div>
          {activeTab === 'details' && <>
            <div className="td-card"><div className="td-card-header">Original Inputs</div><div className="td-card-body">
              <Field label="Transaction ID:" value={original.reference} /><Field label="Transaction Amount:" value={original.amount} amountHelp />
              <Field label="Transaction Type:" value={TRANSACTION_TYPES[original.category]?.label} />
              <Field label="Transaction Date:" value={receiptDate(original.date)} /><Field label="Transaction Time:" value={receiptTime(original.time)} />
              <Field label="Sender Type:" value={roleLabel(original.origin_role)} /><Field label="Recipient Type:" value={roleLabel(original.destination_role)} />
            </div></div>
            <div className="td-card"><div className="td-card-header">Derived Inputs</div><div className="td-card-body">
              <Field label="Hour of the Day:" value={derived.hour_of_day} /><Field label="Day of the Week:" value={weekdayText(derived.day_of_week)} />
              {['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'].map(type => <Field key={type} label={`Transaction is ${type.replaceAll('_', ' ')}:`} value={booleanText(derived[`type_${type}`])} />)}
              <Field label="Transaction Log Amount:" value={derived.log_amount} /><Field label="Transaction Amount is 0:" value={booleanText(derived.is_zero_amount)} />
              <Field label="Origin is Merchant:" value={booleanText(derived.is_merchant_origin)} /><Field label="Destination is Merchant:" value={booleanText(derived.is_merchant_dest)} />
            </div></div>
          </>}
          {activeTab === 'rfSmote' && <ModelResultView {...modelProps('rf_smote')} showGroundTruth={false} />}
          {activeTab === 'benchmark' && <ModelResultView {...modelProps('rf')} showGroundTruth={false} />}
          {activeTab === 'both' && <BothModelsView rfSmote={modelProps('rf_smote')} benchmark={modelProps('rf')} showGroundTruth={false} />}
          {shapModel && <ShapModal model={shapModel} explanation={record.explanation} onRetry={retry} onClose={() => setShapModel(null)} chartApi={receiptApi} chartPresentation={null} />}
        </section>
      <footer className="results-actions receipt-results-actions">
        {error && <p role="alert">{error}</p>}
        <button type="button" className="results-action analyze-button" disabled={working || job.busy}
          onClick={() => action(() => downloadReceipt(id, job.revision))}>Download Results</button>
        <Link className="results-action analyze-button" to={`/results?receipt=${id}`}>Edit Details</Link>
        <button type="button" className="results-action analyze-button" disabled={working || job.busy}
          onClick={() => action(async () => { await clearReceipt(); navigate('/upload') })}>Clear Results</button>
      </footer>
    </main>
  </div>
}
