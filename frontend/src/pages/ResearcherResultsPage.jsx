import "./ResearcherResultsPage.css"
import backbutton from "../assets/backbutton.png"
import stars from "../assets/stars.png"
import { useNavigate } from "react-router-dom"
import SystemResults from "../components/SystemResults"
import SearchBar from "../components/SearchBar"
import TransactionTable from "../components/TransactionTable"
import { useState } from "react"
import StatisticCard from "../components/StatisticCard"
import PercentageDifferenceCard from "../components/PercentageDifference"

const ResearcherResultsPage = () => {
  const navigate = useNavigate()
  const [searchTerm, setSearchTerm] = useState("")

  return (
    <main className="researcher-results-page">
      <header className="researcher-results-header">
        <button type="button" className="back-button" aria-label="Go back" onClick={() => navigate(-1)}>
          <img src={backbutton} alt="Back Button" />
        </button>
      </header>

      <section className="results-overview-section">
        <div className="results-overview-title">
          <h1>Results Overview</h1>
          <img src={stars} alt="Stars Icon" />
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
            />

            <div className="divider"></div>

            <SystemResults
              title="Benchmark RF"
              legitimateRecords="6012120"
              fraudulentRecords="356539"
              fraudRate={0.94}
              titleClass="benchmark-title"
            />
          </div>
          <div className="ground-truth-container">
            <SystemResults
              title="Ground Truth"
              legitimateRecords="6354407"
              fraudulentRecords="8213"
              fraudRate={0.13}
              titleClass="ground-truth-title"
            />
          </div>
        </div>
      </section>

      <section className="transaction-overview-section">
        <div className="results-overview-title">
          <h1>Transaction Records</h1>
          <img src={stars} alt="Stars Icon" />
        </div>

        <SearchBar value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} />
        <TransactionTable searchTerm={searchTerm} />
      </section>

      <section className="performance-statistics-section">
        <div className="results-overview-title">
          <h1>Performance Statistics of Both Models</h1>
          <img src={stars} alt="Stars Icon" />
        </div>

        <div className="model-statistics-container">
          <StatisticCard
            title="RF-SMOTE"
            precision="98.5%"
            recall="99.0%"
            f1="98.7%"
            mcc="98.7%"
            auc="99.1%"
            titleClass="rf-smote-title"
          />
          <StatisticCard
            title="Benchmark RF"
            precision="87.8%"
            recall="75.2%"
            f1="81.0%"
            mcc="80.5%"
            auc="82.4%"
            titleClass="benchmark-title"
          />
        </div>

        <div className="percentage-statistics-container">
          <h1>Percentage Difference</h1>

          <div className="percentage-difference-container">
            <PercentageDifferenceCard percentage="11.49%" label="Precision" />
            <PercentageDifferenceCard percentage="11.49%" label="Recall" />
            <PercentageDifferenceCard percentage="11.49%" label="F1-score" />
            <PercentageDifferenceCard percentage="11.49%" label="MCC" />
            <PercentageDifferenceCard percentage="11.49%" label="PR-AUC" />
          </div>
        </div>

        <div className="mcnemar-statistics-container">
          <h1>McNemar’s Test</h1>

          <div className="mcnemar-content">
            <div className="mcnemar-table-wrapper">
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
              <div className="pvalue-box">
                <p>Resulting p-value:</p>
                <strong>&lt; 0.0001</strong>
              </div>

              <div className="evaluation">
                <p>Evaluation:</p>
                <p>
                  The difference between the benchmark RF model and the RF-SMOTE model is statistically significant.
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
