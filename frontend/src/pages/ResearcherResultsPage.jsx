import "./ResearcherResultsPage.css"
import backbutton from "../assets/backbutton.png"
import stars from "../assets/stars.png"
import { useNavigate } from "react-router-dom"
import SystemResults from "../components/SystemResults"
import SearchBar from "../components/SearchBar"
import FilterBar from "../components/FilterBar"
import TransactionTable from "../components/TransactionTable"
import { useEffect, useState } from "react"
import StatisticCard from "../components/StatisticCard"
import PercentageDifferenceCard from "../components/PercentageDifference"
import ConfusionMatrix from "../components/ConfusionMatrix"
import InfoButton from "../components/InfoButton"

const modelOptions = [
  { value: "both", label: "Both Models" },
  { value: "rfSmote", label: "RF-SMOTE" },
  { value: "benchmark", label: "Benchmark RF" },
]

const outcomeOptions = [
  { value: "all", label: "All Outcomes" },
  { value: "tp", label: "True positive — fraud detected" },
  { value: "fp", label: "False positive — false alarm" },
  { value: "tn", label: "True negative — legitimate recognized" },
  { value: "fn", label: "False negative — fraud missed" },
]

const smoteMetrics = { precision: 98.5, recall: 99.0, f1: 98.7, mcc: 0.987, auc: 99.1 }
const benchmarkMetrics = { precision: 87.8, recall: 75.2, f1: 81.0, mcc: 0.805, auc: 82.4 }
const metricDifference = (key) => {
  const smote = smoteMetrics[key]
  const benchmark = benchmarkMetrics[key]
  const mean = (smote + benchmark) / 2
  if (mean === 0) return "N/A"
  const difference = 100 * (smote - benchmark) / mean
  return `${difference >= 0 ? "+" : ""}${difference.toFixed(2)}%`
}

const ResearcherResultsPage = () => {
  useEffect(() => {
    if (window.particlesJS) {
      window.particlesJS.load("particles-js", "/particles.json", function () {
        console.log("callback - particles.js config loaded")
      })
    }
  }, [])

  const navigate = useNavigate()
  const [searchTerm, setSearchTerm] = useState("")
  const [model, setModel] = useState("both")
  const [outcome, setOutcome] = useState("all")
  const [currentPage, setCurrentPage] = useState(1)
  const handleRequestInfo = () => undefined

  const handleModelChange = (event) => {
    const nextModel = event.target.value
    setModel(nextModel)
    setOutcome("all")
    setCurrentPage(1)
  }

  const handleOutcomeChange = (event) => {
    setOutcome(event.target.value)
    setCurrentPage(1)
  }

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value)
    setCurrentPage(1)
  }

  return (
    <main className="researcher-results-page">
      <div id="particles-js" className="particles-background" aria-hidden="true" />
      <header className="researcher-results-header">
        <button type="button" className="back-button" aria-label="Go back" onClick={() => navigate(-1)}>
          <img src={backbutton} alt="Back Button" />
        </button>
      </header>

      <section className="results-overview-section">
        <div className="results-overview-title with-action">
          <div className="section-heading"><h1>Results Overview</h1><img src={stars} alt="" /></div>
          <button type="button" className="download-button" aria-disabled="true" title="Results download is not available yet">Download results<span className="visually-hidden">; not available yet</span></button>
        </div>

        <div className="file-name-container">
          <b>CSV File Name:</b>
          <span>dataset.csv</span>
        </div>

        <div className="system-results-container">
          <div className="model-result-container">
            <SystemResults
              title="RF-SMOTE"
              legitimateRecords="6308787"
              fraudulentRecords="59872"
              fraudRate={5.6}
              titleClass="rf-smote-title"
              donutClass="rf-smote-donut"
            />

            <div className="divider"></div>

            <SystemResults
              title="Benchmark RF"
              legitimateRecords="6012120"
              fraudulentRecords="356539"
              fraudRate={0.94}
              titleClass="benchmark-title"
              donutClass="benchmark-donut"
            />
          </div>
          <div className="ground-truth-container">
            <SystemResults
              title="Ground Truth"
              legitimateRecords="6354407"
              fraudulentRecords="8213"
              fraudRate={0.13}
              titleClass="ground-truth-title"
              donutClass="ground-truth-donut"
            />
          </div>
        </div>
      </section>

      <section className="transaction-overview-section">
        <div className="results-overview-title">
          <h1>Transaction Records</h1>
          <img src={stars} alt="" />
        </div>

        <div className="search-filter-container">
          <SearchBar value={searchTerm} onChange={handleSearchChange} placeholder="Search Transaction ID" />
          <FilterBar label="Model" value={model} onChange={handleModelChange} options={modelOptions} />
          <FilterBar label="Prediction outcome" value={outcome} onChange={handleOutcomeChange} options={outcomeOptions} disabled={model === "both"} hint="Select one model to filter by prediction outcome." />
        </div>

        <TransactionTable searchTerm={searchTerm} model={model} outcome={outcome} currentPage={currentPage} onPageChange={setCurrentPage} />
      </section>

      <section className="confusion-matrices-container">
        <div className="results-overview-title">
          <h1>Confusion Matrices</h1>
          <img src={stars} alt="" />
        </div>

        <ConfusionMatrix
          title="RF-SMOTE Model"
          truePositive={400000}
          falseNegative={400000}
          falsePositive={400000}
          trueNegative={400000}
        />

        <ConfusionMatrix
          title="Benchmark RF Model"
          truePositive={400000}
          falseNegative={400000}
          falsePositive={400000}
          trueNegative={400000}
        />
      </section>

      <section className="performance-statistics-section">
        <div className="results-overview-title">
          <h1>Performance Statistics</h1>
          <img src={stars} alt="" />
        </div>

        <div className="model-statistics-container">
          <StatisticCard
            title="RF-SMOTE"
            precision={`${smoteMetrics.precision.toFixed(1)}%`}
            recall={`${smoteMetrics.recall.toFixed(1)}%`}
            f1={`${smoteMetrics.f1.toFixed(1)}%`}
            mcc={smoteMetrics.mcc.toFixed(3)}
            auc={`${smoteMetrics.auc.toFixed(1)}%`}
            onRequestInfo={handleRequestInfo}
          />
          <StatisticCard
            title="Benchmark RF"
            precision={`${benchmarkMetrics.precision.toFixed(1)}%`}
            recall={`${benchmarkMetrics.recall.toFixed(1)}%`}
            f1={`${benchmarkMetrics.f1.toFixed(1)}%`}
            mcc={benchmarkMetrics.mcc.toFixed(3)}
            auc={`${benchmarkMetrics.auc.toFixed(1)}%`}
            onRequestInfo={handleRequestInfo}
          />
        </div>

        <div className="percentage-main-container">
          <div className="percentage-statistics-title-container">
            <InfoButton topic="Percentage Difference" onRequestInfo={handleRequestInfo} />
            <h3>Percentage Difference</h3>
          </div>
          <div className="percentage-statistics-container">
            <div className="percentage-statistics-guide-container">
              <h3>Percentage Guide:</h3>
              <p>&#8226; A positive value indicates a better performance in favor of RF-SMOTE model.</p>
              <p>&#8226; A negative value indicates a better performance in favor of Benchmark RF model.</p>
            </div>

            <div className="percentage-difference-container">
              <PercentageDifferenceCard percentage={metricDifference("precision")} label="Precision" />
              <PercentageDifferenceCard percentage={metricDifference("recall")} label="Recall" />
              <PercentageDifferenceCard percentage={metricDifference("f1")} label="F1-Score" />
              <PercentageDifferenceCard percentage={metricDifference("auc")} label="PR-AUC" />
              <PercentageDifferenceCard percentage={metricDifference("mcc")} label="MCC" />
            </div>
          </div>
        </div>
      </section>

      <section className="statistical-significance-container">
        <div className="results-overview-title">
          <h1>Statistical Significance</h1>
          <img src={stars} alt="" />
        </div>

        <div className="mcnemar-statistics-title-container">
          <InfoButton topic="McNemar's Test" onRequestInfo={handleRequestInfo} />
          <h3>McNemar’s Test</h3>
        </div>
        <div className="mcnemar-statistics-container">
          <div className="mcnemar-content">
            <div className="mcnemar-table-wrapper">
              <div className="mcnemar-label"><h4>Contingency Table:</h4><InfoButton topic="Contingency Table" onRequestInfo={handleRequestInfo} /></div>
              <div className="mcnemar-table">
                <div className="empty-cell"></div>
                <div className="table-header">RF-SMOTE Model Correct</div>
                <div className="table-header">RF-SMOTE Model Incorrect</div>

                <div className="row-header">RF Model Correct</div>
                <div className="table-value">6005000</div>
                <div className="table-value">125000</div>

                <div className="row-header">RF Model Incorrect</div>
                <div className="table-value">345000</div>
                <div className="table-value">345000</div>
              </div>
            </div>

            <div className="mcnemar-result">
              <div className="mcnemar-label"><h4>P-Value:</h4><InfoButton topic="P-Value" onRequestInfo={handleRequestInfo} /></div>
              <div className="pvalue-box">
                <strong>2.600e-100</strong>
                <span>&lt; 0.05</span>
              </div>

              <h4>Interpretation:</h4>
              <div className="interpretation">
                <p>
                  Reject the null hypothesis. The models have different classification error rates on this evaluation set.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}

export default ResearcherResultsPage
