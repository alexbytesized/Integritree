import "./TransactionDetailsPage.css"
import "./ResearcherResultsPage.css"
import stars from "../assets/stars.png"
import ReturnButton from "../components/ReturnButton"
import ReceiptParticles from "../components/ReceiptParticles"
import TabBar from "../components/TabBar"
import ModelResultView from "../components/ModelResultView"
import BothModelsView from "../components/BothModelsView"
import { useEffect, useState } from "react"
import { Link, useParams, useSearchParams } from "react-router-dom"
import { api, analysisKey, label, outcomeName } from "../researchApi"
import ShapModal from '../components/ShapModal'
import { MODEL_NAMES, booleanText, weekdayText } from '../researchDisplay'
import ShapSummary from '../components/ShapSummary'

const TABS = [
  { key: "details",   label: "Transaction Details" },
  { key: "rfSmote",   label: "RF-SMOTE" },
  { key: "benchmark", label: "Benchmark RF" },
  { key: "both",      label: "Both Models" },
]

const Field = ({ label, value }) => (
  <div className="td-field-row">
    <span className="td-field-label">{label}</span>
    <div className="td-field-value">{value == null || value === '' ? 'Not supplied' : String(value)}</div>
  </div>
)

const TransactionDetailsPage = () => {
  const { id: number } = useParams()
  const [params] = useSearchParams()
  const analysis = params.get('analysis') || sessionStorage.getItem(analysisKey)
  return <TransactionRecord key={`${analysis}-${number}`} analysis={analysis} number={number} returnQuery={params.get('return')} />
}

const TransactionRecord = ({ analysis, number, returnQuery }) => {
  const [activeTab, setActiveTab] = useState('details')
  const [shapModel, setShapModel] = useState(null)
  const back = `/researcher?${returnQuery || `analysis=${analysis}`}`
  const [record, setRecord] = useState(null)
  const [error, setError] = useState('')
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    if (!analysis) return
    let live = true
    let timer
    let requested = false
    const poll = async () => {
      try {
        const data = await api(`/analyses/${analysis}/records/${number}`)
        if (!live) return
        setRecord(data)
        if (!requested && (data.explanation.status === 'not_requested' || (revision > 0 && data.explanation.status === 'failed'))) {
          requested = true
          await api(`/analyses/${analysis}/records/${number}/explanation`, { method: 'POST' })
          if (live) timer = setTimeout(poll, 1200)
        } else if (data.explanation.status === 'pending') timer = setTimeout(poll, 1200)
      } catch (err) { if (live) setError(err.message) }
    }
    poll()
    return () => { live = false; clearTimeout(timer) }
  }, [analysis, number, revision])

  if (!record) return <main className="researcher-results-page"><h1>Transaction details</h1><p role="status">{error || (analysis ? 'Loading...' : 'Session expired. Upload a CSV again.')}</p><Link to="/researcher-upload">Return to upload</Link></main>
  const modelProps = (name) => ({
    prediction: label(record[name].predicted_label), riskScore: record[name].score * 100,
    groundTruth: label(record.actual_label), outcome: outcomeName(record[name].predicted_label, record.actual_label),
    threshold: record.threshold * 100,
    modelName: MODEL_NAMES[name],
    onOpenShap: () => setShapModel(name),
    shapSummary: <ShapSummary explanation={record.explanation} model={name} result={record[name]} />,
  })
  const original = record.original
  const derived = record.derived
  const originalValue = value => record.input_format === 'prepared'
    ? 'N/A (file uploaded is already preprocessed)' : value
  const transaction = { ...original, id: record.row_number, isFraud: record.actual_label,
    oldBalanceOrig: original.oldbalanceOrg, newBalanceOrig: original.newbalanceOrig,
    oldBalanceDest: original.oldbalanceDest, newBalanceDest: original.newbalanceDest,
    hourOfDay: derived.hour_of_day, dayOfWeek: derived.day_of_week, isCashIn: derived.type_CASH_IN,
    isCashOut: derived.type_CASH_OUT, isDebit: derived.type_DEBIT, isPayment: derived.type_PAYMENT,
    isTransfer: derived.type_TRANSFER, logAmount: derived.log_amount, amountIsZero: derived.is_zero_amount,
    origIsMerchant: derived.is_merchant_origin, destIsMerchant: derived.is_merchant_dest,
    rfSmote: modelProps('rf_smote'), benchmark: modelProps('rf') }
  return (
    <div className="researcher-results-wrapper">
      <ReceiptParticles />
      <ReturnButton to={back} />
      <main className="researcher-results-page">
        {error && <p role="alert">{error}</p>}

        {/* Heading + Tab bar */}
        <section className="results-overview-section">
          <div className="results-overview-title">
            <div className="section-heading">
              <h1>Results Overview</h1>
              <img src={stars} alt="" />
            </div>
          </div>
          <TabBar tabs={TABS} active={activeTab} onChange={tab => { setShapModel(null); setActiveTab(tab) }} />
        </section>

        {/* Tab content */}
        <section className="td-content-section">

          {/* Shared heading visible on all tabs */}
          <div className="td-record-heading">
            <h2 className="td-record-title">Transaction Record</h2>
            <p className="td-record-id">
              <span className="td-record-id-label">Transaction ID:</span> {transaction.id}
            </p>
          </div>

          {/* Transaction Details tab */}
          {activeTab === "details" && (
            <>
              <div className="td-card">
                <div className="td-card-header">Original Inputs</div>
                <div className="td-card-body">
                  <Field label="Step:"                      value={originalValue(transaction.step)} />
                  <Field label="Transaction Type:"          value={originalValue(transaction.type)} />
                  <Field label="Transaction Amount:"        value={originalValue(transaction.amount)} />
                  <Field label="Origin Name:"               value={originalValue(transaction.nameOrig)} />
                  <Field label="Origin Old Balance:"        value={originalValue(transaction.oldBalanceOrig)} />
                  <Field label="Origin New Balance:"        value={originalValue(transaction.newBalanceOrig)} />
                  <Field label="Destination Name:"          value={originalValue(transaction.nameDest)} />
                  <Field label="Destination Old Balance:"   value={originalValue(transaction.oldBalanceDest)} />
                  <Field label="Destination New Balance:"   value={originalValue(transaction.newBalanceDest)} />
                  <Field label="Transaction is Fraudulent:" value={booleanText(transaction.isFraud)} />
                </div>
              </div>

              <div className="td-card">
                <div className="td-card-header">Derived Inputs</div>
                <div className="td-card-body">
                  <Field label="Hour of the Day"            value={transaction.hourOfDay} />
                  <Field label="Day of the Week"            value={weekdayText(transaction.dayOfWeek)} />
                  <Field label="Transaction is Cash In:"    value={booleanText(transaction.isCashIn)} />
                  <Field label="Transaction is Cash Out:"   value={booleanText(transaction.isCashOut)} />
                  <Field label="Transaction is Debit:"      value={booleanText(transaction.isDebit)} />
                  <Field label="Transaction is Payment:"    value={booleanText(transaction.isPayment)} />
                  <Field label="Transaction is Transfer:"   value={booleanText(transaction.isTransfer)} />
                  <Field label="Transaction Log Amount:"    value={transaction.logAmount} />
                  <Field label="Transaction Amount is 0:"   value={booleanText(transaction.amountIsZero)} />
                  <Field label="Origin is Merchant"         value={booleanText(transaction.origIsMerchant)} />
                  <Field label="Destination is Merchant:"   value={booleanText(transaction.destIsMerchant)} />
                </div>
              </div>
            </>
          )}

          {/* RF-SMOTE tab */}
          {activeTab === "rfSmote" && (
            <ModelResultView {...transaction.rfSmote} />
          )}

          {/* Benchmark RF tab */}
          {activeTab === "benchmark" && (
            <ModelResultView {...transaction.benchmark} />
          )}

          {/* Both Models tab */}
          {activeTab === "both" && (
            <BothModelsView
              rfSmote={transaction.rfSmote}
              benchmark={transaction.benchmark}
            />
          )}

          {shapModel && <ShapModal model={shapModel} explanation={record.explanation}
            onRetry={() => setRevision(v => v + 1)} onClose={() => setShapModel(null)} />}

        </section>

      </main>
    </div>
  )
}

export default TransactionDetailsPage
