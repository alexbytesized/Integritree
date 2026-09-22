import "./ResearcherResultsPage.css"
import backbutton from "../assets/backbutton.png"
import stars from "../assets/stars.png"
import { useNavigate } from "react-router-dom"
import SystemResults from "../components/SystemResults"
import SearchBar from "../components/SearchBar"
import FilterBar from "../components/FilterBar"
import TransactionTable from "../components/TransactionTable"
import { useState } from "react"
import StatisticCard from "../components/StatisticCard"
import PercentageDifferenceCard from "../components/PercentageDifference"
import ConfusionMatrix from "../components/ConfusionMatrix"
import React, { useEffect } from "react"

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
  const [filter, setFilter] = useState("all")

  return (
    <main className="researcher-results-page">
      <div id="particles-js" className="particles-background" aria-hidden="true" />
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
          <img src={stars} alt="Stars Icon" />
        </div>

        <div className="search-filter-container">
          <SearchBar value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} />
          <FilterBar value={filter} onChange={(e) => setFilter(e.target.value)} />
        </div>

        <TransactionTable searchTerm={searchTerm} />
      </section>

      <section className="confusion-matrices-container">
        <div className="results-overview-title">
          <h1>Confusion Matrices</h1>
          <img src={stars} alt="Stars Icon" />
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

        <div className="percentage-main-container">
          <div className="percentage-statistics-title-container">
            <h1>PERCENTAGE DIFFERENCE</h1>
          </div>
          <div className="percentage-statistics-container">
            <div className="percentage-statistics-guide-container">
              <h3>Percentage Guide:</h3>
              <p>&#8226; A positive value indicates a better performance in favor of RF-SMOTE model.</p>
              <p>&#8226; A negative value indicates a better performance in favor of Benchmark RF model.</p>
            </div>

            <div className="percentage-difference-container">
              <PercentageDifferenceCard percentage="11.49%" label="Precision" />
              <PercentageDifferenceCard percentage="11.49%" label="Recall" />
              <PercentageDifferenceCard percentage="11.49%" label="F1-score" />
              <PercentageDifferenceCard percentage="11.49%" label="MCC" />
              <PercentageDifferenceCard percentage="11.49%" label="PR-AUC" />
            </div>
          </div>
        </div>
      </section>

      <section className="statistical-significance-container">
        <div className="results-overview-title">
          <h1>Statistical Significance</h1>
          <img src={stars} alt="Stars Icon" />
        </div>

        <div className="mcnemar-statistics-title-container">
          <h1>MCNEMAR'S TEST</h1>
        </div>
        <div className="mcnemar-statistics-container">
          <h2>Contingency Table:</h2>
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
              <h2>P-value:</h2>
              <div className="pvalue-box">
                <strong>2.600e-100</strong>
                <span>&lt; 0.05</span>
              </div>

              <h2>Interpretation:</h2>
              <div className="interpretation">
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
