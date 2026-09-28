import "./TransactionDetailsPage.css"
import "./ResearcherResultsPage.css"
import stars from "../assets/stars.png"
import ReturnButton from "../components/ReturnButton"
import TabBar from "../components/TabBar"
import ModelResultView from "../components/ModelResultView"
import BothModelsView from "../components/BothModelsView"
import { useEffect, useState } from "react"

const TABS = [
  { key: "details",   label: "Transaction Details" },
  { key: "rfSmote",   label: "RF-SMOTE" },
  { key: "benchmark", label: "Benchmark RF" },
  { key: "both",      label: "Both Models" },
]

// Placeholder transaction data — replace with API/router state
const transaction = {
  id: "12345678",
  step: 1,
  type: "Transfer",
  amount: 9839.64,
  nameOrig: "C1231006815",
  oldBalanceOrig: 170136,
  newBalanceOrig: 160296.36,
  nameDest: "M1979787155",
  oldBalanceDest: 0,
  newBalanceDest: 0,
  isFraud: true,
  hourOfDay: 1,
  dayOfWeek: 0,
  isCashIn: false,
  isCashOut: false,
  isDebit: false,
  isPayment: true,
  isTransfer: false,
  logAmount: 25,
  amountIsZero: false,
  origIsMerchant: false,
  destIsMerchant: true,
  // Model results — replace with API data
  rfSmote: {
    prediction: "Fraudulent",
    riskScore: 75,
    groundTruth: "Fraudulent",
    outcome: "True Positive",
  },
  benchmark: {
    prediction: "Fraudulent",
    riskScore: 68,
    groundTruth: "Fraudulent",
    outcome: "True Positive",
  },
}

const Field = ({ label, value }) => (
  <div className="td-field-row">
    <span className="td-field-label">{label}</span>
    <div className="td-field-value">{String(value)}</div>
  </div>
)

const TransactionDetailsPage = () => {
  const [activeTab, setActiveTab] = useState("details")

  useEffect(() => {
    if (window.particlesJS) {
      window.particlesJS.load("particles-js", "/particles.json", function () {})
    }
    return () => {
      if (window.pJSDom && window.pJSDom.length > 0) {
        window.pJSDom.forEach((entry) => {
          if (entry.pJS?.fn?.vendors?.destroypJS) entry.pJS.fn.vendors.destroypJS()
        })
        window.pJSDom = []
      }
    }
  }, [])

  return (
    <div className="researcher-results-wrapper">
      <div id="particles-js" className="particles-background" aria-hidden="true" />
      <ReturnButton to="/researcher" />
      <main className="researcher-results-page">

        {/* Heading + Tab bar */}
        <section className="results-overview-section">
          <div className="results-overview-title">
            <div className="section-heading">
              <h1>Results Overview</h1>
              <img src={stars} alt="" />
            </div>
          </div>
          <TabBar tabs={TABS} active={activeTab} onChange={setActiveTab} />
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
                  <Field label="Step:"                      value={transaction.step} />
                  <Field label="Transaction Type:"          value={transaction.type} />
                  <Field label="Transaction Amount:"        value={transaction.amount} />
                  <Field label="Origin Name:"               value={transaction.nameOrig} />
                  <Field label="Origin Old Balance:"        value={transaction.oldBalanceOrig} />
                  <Field label="Origin New Balance:"        value={transaction.newBalanceOrig} />
                  <Field label="Destination Name:"          value={transaction.nameDest} />
                  <Field label="Destination Old Balance:"   value={transaction.oldBalanceDest} />
                  <Field label="Destination New Balance:"   value={transaction.newBalanceDest} />
                  <Field label="Transaction is Fraudulent:" value={transaction.isFraud} />
                </div>
              </div>

              <div className="td-card">
                <div className="td-card-header">Derived Inputs</div>
                <div className="td-card-body">
                  <Field label="Hour of the Day"            value={transaction.hourOfDay} />
                  <Field label="Day of the Week"            value={transaction.dayOfWeek} />
                  <Field label="Transaction is Cash In:"    value={transaction.isCashIn} />
                  <Field label="Transaction is Cash Out:"   value={transaction.isCashOut} />
                  <Field label="Transaction is Debit:"      value={transaction.isDebit} />
                  <Field label="Transaction is Payment:"    value={transaction.isPayment} />
                  <Field label="Transaction is Transfer:"   value={transaction.isTransfer} />
                  <Field label="Transaction Log Amount:"    value={transaction.logAmount} />
                  <Field label="Transaction Amount is 0:"   value={transaction.amountIsZero} />
                  <Field label="Origin is Merchant"         value={transaction.origIsMerchant} />
                  <Field label="Destination is Merchant:"   value={transaction.destIsMerchant} />
                </div>
              </div>
            </>
          )}

          {/* RF-SMOTE tab */}
          {activeTab === "rfSmote" && (
            <ModelResultView
              prediction={transaction.rfSmote.prediction}
              riskScore={transaction.rfSmote.riskScore}
              groundTruth={transaction.rfSmote.groundTruth}
              outcome={transaction.rfSmote.outcome}
            />
          )}

          {/* Benchmark RF tab */}
          {activeTab === "benchmark" && (
            <ModelResultView
              prediction={transaction.benchmark.prediction}
              riskScore={transaction.benchmark.riskScore}
              groundTruth={transaction.benchmark.groundTruth}
              outcome={transaction.benchmark.outcome}
            />
          )}

          {/* Both Models tab */}
          {activeTab === "both" && (
            <BothModelsView
              rfSmote={transaction.rfSmote}
              benchmark={transaction.benchmark}
            />
          )}

        </section>

      </main>
    </div>
  )
}

export default TransactionDetailsPage
