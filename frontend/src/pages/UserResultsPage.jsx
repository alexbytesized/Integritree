import { useEffect, useState } from "react"
import { useLocation } from "react-router-dom"

import stars from "../assets/stars.png"
import ReturnButton from "../components/ReturnButton"
import ScreenshotPreview from "../components/ScreenshotPreview"
import TabBar from "../components/TabBar"
import ModelResultView from "../components/ModelResultView"
import BothModelsView from "../components/BothModelsView"

import "./UserResultsPage.css"

const TABS = [
  { key: "details",   label: "Transaction Details" },
  { key: "rfSmote",   label: "RF-SMOTE" },
  { key: "benchmark", label: "Benchmark RF" },
  { key: "both",      label: "Both Models" },
]

const Field = ({ label, value }) => (
  <div className="td-field-row">
    <span className="td-field-label">{label}</span>
    <div className="td-field-value">{String(value)}</div>
  </div>
)

const UserResultsPage = () => {
  const location = useLocation()

  const {
    fileName = "screenshot.png",
    imageUrl = null,
    transactionData = {
      transactionId: "12345678",
      amount: "5000.00",
      transactionType: "Transfer",
      transactionDate: "September 14, 2026",
      transactionTime: "3:15 AM",
      senderType: "Customer",
      recipient: "Customer",
    }
  } = location.state || {}

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

  // Model results for predictions
  const modelResults = {
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

  return (
    <div className="researcher-results-wrapper">
      <div id="particles-js" className="particles-background" aria-hidden="true" />
      <ReturnButton to={-1} />

      <main className="user-details-page user-results-page">
        <section className="user-results-overview">
          <div className="user-overview-title">
            <div className="section-heading">
              <h1>Results Overview</h1>
              <img src={stars} alt="" />
            </div>
          </div>

          <div className="user-results-content">
            {/* Screenshot preview */}
            <ScreenshotPreview fileName={fileName} imageUrl={imageUrl} />

            {/* Tab navigation */}
            <div className="user-results-tabs-wrapper">
              <TabBar tabs={TABS} active={activeTab} onChange={setActiveTab} />
            </div>

            {/* Tab content matching researcher */}
            <section className="td-content-section">
              {/* Shared Record Heading */}
              <div className="td-record-heading">
                <h2 className="td-record-title">Transaction Record</h2>
                <p className="td-record-id">
                  <span className="td-record-id-label">Transaction ID:</span> {transactionData.transactionId || "12345678"}
                </p>
              </div>

              {/* Tab contents */}
              {activeTab === "details" && (
                <div className="td-card">
                  <div className="td-card-header">Transaction Details</div>
                  <div className="td-card-body">
                    <Field label="Transaction ID:"      value={transactionData.transactionId || "12345678"} />
                    <Field label="Transaction Amount:"  value={transactionData.amount || "₱5,000.00"} />
                    <Field label="Transaction Type:"    value={transactionData.transactionType || "Transfer"} />
                    <Field label="Transaction Date:"    value={transactionData.transactionDate || "September 14, 2026"} />
                    <Field label="Transaction Time:"    value={transactionData.transactionTime || "3:15 AM"} />
                    <Field label="Sender Type:"         value={transactionData.senderType || "Customer"} />
                    <Field label="Recipient Type:"      value={transactionData.recipient || "Customer"} />
                  </div>
                </div>
              )}

              {activeTab === "rfSmote" && (
                <ModelResultView
                  prediction={modelResults.rfSmote.prediction}
                  riskScore={modelResults.rfSmote.riskScore}
                  groundTruth={modelResults.rfSmote.groundTruth}
                  outcome={modelResults.rfSmote.outcome}
                />
              )}

              {activeTab === "benchmark" && (
                <ModelResultView
                  modelName="Benchmark RF"
                  prediction={modelResults.benchmark.prediction}
                  riskScore={modelResults.benchmark.riskScore}
                  groundTruth={modelResults.benchmark.groundTruth}
                  outcome={modelResults.benchmark.outcome}
                />
              )}

              {activeTab === "both" && (
                <BothModelsView
                  rfSmote={modelResults.rfSmote}
                  benchmark={modelResults.benchmark}
                />
              )}
            </section>
          </div>
        </section>
      </main>
    </div>
  )
}

export default UserResultsPage
