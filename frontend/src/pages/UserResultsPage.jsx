import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import stars from '../assets/stars.png'
import ReturnButton from '../components/ReturnButton'
import ScreenshotPreview from '../components/ScreenshotPreview'
import TabBar from '../components/TabBar'
import ModelResultView from '../components/ModelResultView'
import BothModelsView from '../components/BothModelsView'
import ShapModal from '../components/ShapModal'
import { MODEL_NAMES, booleanText, weekdayText, shapSummary } from '../researchDisplay'
import { receiptApi, receiptKey, clearReceipt, downloadReceipt } from '../receiptApi'
import { useReceipt, useReceiptImage } from '../useReceipt'
import { WORKFLOWS } from '../receiptDisplay'
import './TransactionDetailsPage.css'
import './UploadDetails.css'
import './UserResultsPage.css'
import './ReceiptFlow.css'

const TABS = [{ key: 'details', label: 'Transaction Details' }, { key: 'rfSmote', label: 'RF-SMOTE' },
  { key: 'benchmark', label: 'Benchmark RF' }, { key: 'both', label: 'Both Models' }]
const Field = ({ label, value }) => <div className="td-field-row"><span className="td-field-label">{label}</span><div className="td-field-value">{value == null || value === '' ? 'Not supplied' : String(value)}</div></div>

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
    <main className="user-details-page user-results-page receipt-page">
      <section className="user-results-overview">
        <div className="user-overview-title section-heading"><h1>Results Overview</h1><img src={stars} alt="" /></div>
        <p>Experimental transaction predictions. These scores do not verify receipt authenticity or establish real-world fraud.</p>
        {(error || loadError) && <p role="alert">{error || loadError}</p>}
        <div className="receipt-actions">
          <Link to={`/results?receipt=${id}`}>Edit confirmed details</Link>
          <button disabled={working || job.busy} onClick={() => action(() => downloadReceipt(id, job.revision))}>Download results</button>
          <button disabled={working || job.busy} onClick={() => action(async () => { await clearReceipt(id); navigate('/upload') })}>Clear results</button>
        </div>
        <ScreenshotPreview fileName={job.filename} imageUrl={image} />
        <div className="user-results-tabs-wrapper"><TabBar tabs={TABS} active={activeTab} onChange={setActiveTab} /></div>
        <section className="td-content-section">
          <div className="td-record-heading"><h2 className="td-record-title">Transaction Record</h2><p className="td-record-id">Receipt analysis</p></div>
          {activeTab === 'details' && <>
            <div className="td-card"><div className="td-card-header">Original Inputs</div><div className="td-card-body">
              <Field label="Reference:" value={original.reference} /><Field label="Principal Amount (PHP):" value={original.amount} />
              <Field label="GCash Workflow:" value={WORKFLOWS[original.workflow]?.label} /><Field label="Transaction Type:" value={original.category} />
              <Field label="Transaction Date (Manila):" value={original.date} /><Field label="Transaction Time (Manila):" value={original.time} />
              <Field label="Origin Role:" value={original.origin_role.replaceAll('_', ' ')} /><Field label="Destination Role:" value={original.destination_role.replaceAll('_', ' ')} />
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
        <p className="receipt-retention">Download before clearing results or stopping/restarting the backend. Closing the tab alone does not delete the receipt.</p>
      </section>
    </main>
  </div>
}
