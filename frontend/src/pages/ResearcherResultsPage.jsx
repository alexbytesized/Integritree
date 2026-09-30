import "./ResearcherResultsPage.css"
import stars from "../assets/stars.png"
import ReturnButton from "../components/ReturnButton"
import SystemResults from "../components/SystemResults"
import SearchBar from "../components/SearchBar"
import FilterBar from "../components/FilterBar"
import TransactionTable from "../components/TransactionTable"
import { useCallback, useEffect, useState } from "react"
import StatisticCard from "../components/StatisticCard"
import PercentageDifferenceCard from "../components/PercentageDifference"
import ConfusionMatrix from "../components/ConfusionMatrix"
import InfoButton from "../components/InfoButton"
import InfoModal from "../components/InfoModal"
import ResearchAnalysisLoading from "../components/ResearchAnalysisLoading"
import { Link, useSearchParams } from "react-router-dom"
import { api, analysisKey, downloadAnalysis, metric } from "../researchApi"

const modelOptions = [
  { value: "both", label: "Both Models" },
  { value: "rfSmote", label: "RF-SMOTE" },
  { value: "benchmark", label: "Benchmark RF" },
]

const outcomeOptions = [
  { value: "all", label: "All Outcomes" },
  { value: "tp", label: "True Positive" },
  { value: "fp", label: "False Positive" },
  { value: "tn", label: "True Negative" },
  { value: "fn", label: "False Negative" },
]

const ResearcherResultsPage = () => {
  useEffect(() => {
    if (window.particlesJS) {
      window.particlesJS.load("particles-js", "/particles.json", function () {
        console.log("callback - particles.js config loaded")
      })
    }

    return () => {
      if (window.pJSDom && window.pJSDom.length > 0) {
        window.pJSDom.forEach((entry) => {
          if (entry.pJS && entry.pJS.fn && entry.pJS.fn.vendors && entry.pJS.fn.vendors.destroypJS) {
            entry.pJS.fn.vendors.destroypJS()
          }
        })
        window.pJSDom = []
      }
    }
  }, [])

  const [params, setParams] = useSearchParams()
  const id = params.get('analysis') || sessionStorage.getItem(analysisKey)
  const searchTerm = params.get('search') || ''
  const model = params.get('model') || 'both'
  const outcome = params.get('outcome') || 'all'
  const currentPage = Math.max(1, Number(params.get('page')) || 1)
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const [downloading, setDownloading] = useState(false)
  const [activeInfoTopic, setActiveInfoTopic] = useState(null)
  const closeInfo = useCallback(() => setActiveInfoTopic(null), [])
  const change = (values) => setParams({ analysis: id, search: searchTerm, model, outcome, page: currentPage, ...values }, { replace: true })
  const handleModelChange = (event) => change({ model: event.target.value, outcome: 'all', page: 1 })
  const handleOutcomeChange = (event) => change({ outcome: event.target.value, page: 1 })
  const handleSearchChange = (event) => change({ search: event.target.value, page: 1 })
  const setCurrentPage = (page) => change({ page })
  useEffect(() => {
    if (!id) return
    let live = true
    let timer
    const poll = async () => {
      try {
        const data = await api(`/analyses/${id}`)
        if (!live) return
        setJob(data)
        if (!['complete', 'failed'].includes(data.status)) timer = setTimeout(poll, 1000)
      } catch (err) { if (live) setError(err.message) }
    }
    poll()
    return () => { live = false; clearTimeout(timer) }
  }, [id])
  const download = async () => {
    setDownloading(true)
    setError('')
    try { await downloadAnalysis(id) } catch (err) { setError(err.message) }
    finally { setDownloading(false) }
  }
  const clear = async () => {
    try {
      await api(`/analyses/${id}`, { method: 'DELETE' })
      sessionStorage.removeItem(analysisKey)
      window.location.assign('/researcher-upload')
    } catch (err) { setError(err.message) }
  }
  if (id && !error && !['complete', 'failed'].includes(job?.status)) {
    return <ResearchAnalysisLoading status={job?.status} rowsProcessed={job?.rows_processed ?? 0} />
  }
  if (!id || job?.status !== 'complete') return <main className="researcher-results-page">
    <h1>CSV analysis</h1>
    {error && <p role="alert">{error}</p>}
    {job?.status === 'failed' ? <><p role="alert">{job.error}</p><pre>{job.issues && JSON.stringify(job.issues, null, 2)}</pre></>
      : <p>Upload a CSV to start a new analysis.</p>}
    <Link to="/researcher-upload">Return to upload / retry</Link>
    {job?.status === 'failed' && <button onClick={clear}>Clear failed analysis</button>}
  </main>
  const evaluation = job.evaluation
  const sm = evaluation.models.rf_smote
  const rf = evaluation.models.rf
  const stats = (model) => ({ precision: metric(model.metrics.precision), recall: metric(model.metrics.recall),
    f1: metric(model.metrics.f1), mcc: metric(model.metrics.mcc, true), auc: metric(model.metrics.pr_auc) })
  const smoteMetrics = stats(sm)
  const benchmarkMetrics = stats(rf)
  const metricDifference = (key) => {
    const item = evaluation.descriptive_comparisons[key === 'auc' ? 'pr_auc' : key]
    return item.value == null ? `N/A (${item.reason})` : `${item.value >= 0 ? '+' : ''}${item.value.toFixed(3)}${item.units === 'percent' ? '%' : ' coefficient'}`
  }
  const test = evaluation.statistical_test
  const rate = (count) => 100 * count / job.rows_processed
  return (
    <div className="researcher-results-wrapper">
      <div id="particles-js" className="particles-background" aria-hidden="true" />
      <ReturnButton to="/researcher-upload" />
      <main className="researcher-results-page">

      <section className="results-overview-section">
        <div className="results-overview-title">
          <div className="section-heading"><h1>Results Overview</h1><img src={stars} alt="" /></div>
        </div>

        <div className="file-name-container">
          <b>CSV File Name:</b>
          <span>{job.filename}</span>
        </div>

        <div className="system-results-container">
          <div className="model-result-container">
            <SystemResults
              title="RF-SMOTE"
              legitimateRecords={sm.confusion_matrix.tn + sm.confusion_matrix.fn}
              fraudulentRecords={sm.confusion_matrix.tp + sm.confusion_matrix.fp}
              fraudRate={rate(sm.confusion_matrix.tp + sm.confusion_matrix.fp)}
              titleClass="rf-smote-title"
              donutClass="rf-smote-donut"
            />

            <div className="divider"></div>

            <SystemResults
              title="Benchmark RF"
              legitimateRecords={rf.confusion_matrix.tn + rf.confusion_matrix.fn}
              fraudulentRecords={rf.confusion_matrix.tp + rf.confusion_matrix.fp}
              fraudRate={rate(rf.confusion_matrix.tp + rf.confusion_matrix.fp)}
              titleClass="benchmark-title"
              donutClass="benchmark-donut"
            />
          </div>
          <div className="ground-truth-container">
            <SystemResults
              title="Ground Truth"
              legitimateRecords={job.rows_processed - rf.actual_fraud}
              fraudulentRecords={rf.actual_fraud}
              fraudRate={rate(rf.actual_fraud)}
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
          <FilterBar label="Prediction outcome" value={outcome} onChange={handleOutcomeChange} options={outcomeOptions} disabled={model === "both"} />
        </div>

        <TransactionTable analysisId={id} returnQuery={params.toString()} searchTerm={searchTerm} model={model} outcome={outcome} currentPage={currentPage} onPageChange={setCurrentPage} />
      </section>

      <section className="confusion-matrices-container">
        <div className="results-overview-title">
          <h1>Confusion Matrices</h1>
          <img src={stars} alt="" />
        </div>

        <ConfusionMatrix
          title="RF-SMOTE Model"
          truePositive={sm.confusion_matrix.tp}
          falseNegative={sm.confusion_matrix.fn}
          falsePositive={sm.confusion_matrix.fp}
          trueNegative={sm.confusion_matrix.tn}
        />

        <ConfusionMatrix
          title="Benchmark RF Model"
          truePositive={rf.confusion_matrix.tp}
          falseNegative={rf.confusion_matrix.fn}
          falsePositive={rf.confusion_matrix.fp}
          trueNegative={rf.confusion_matrix.tn}
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
            precision={smoteMetrics.precision}
            recall={smoteMetrics.recall}
            f1={smoteMetrics.f1}
            mcc={smoteMetrics.mcc}
            auc={smoteMetrics.auc}
            onRequestInfo={setActiveInfoTopic}
          />
          <StatisticCard
            title="Benchmark RF"
            precision={benchmarkMetrics.precision}
            recall={benchmarkMetrics.recall}
            f1={benchmarkMetrics.f1}
            mcc={benchmarkMetrics.mcc}
            auc={benchmarkMetrics.auc}
            onRequestInfo={setActiveInfoTopic}
          />
        </div>

        <div className="percentage-main-container">
          <div className="percentage-statistics-title-container">
            <InfoButton topic="Percentage Difference" onRequestInfo={setActiveInfoTopic} />
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
          <InfoButton topic="McNemar's Test" onRequestInfo={setActiveInfoTopic} />
          <h3>McNemar’s Test</h3>
        </div>
        <div className="mcnemar-statistics-container">
          <div className="mcnemar-content">
            <div className="mcnemar-table-wrapper">
              <h4 className="mcnemar-label">Contingency Table:</h4>
              <div className="mcnemar-table">
                <div className="empty-cell"></div>
                <div className="table-header">RF-SMOTE Model Correct</div>
                <div className="table-header">RF-SMOTE Model Incorrect</div>

                <div className="row-header">RF Model Correct</div>
                <div className="table-value">{test.table.both_correct}</div>
                <div className="table-value">{test.table.rf_only_correct}</div>

                <div className="row-header">RF Model Incorrect</div>
                <div className="table-value">{test.table.smote_only_correct}</div>
                <div className="table-value">{test.table.both_incorrect}</div>
              </div>
            </div>

            <div className="mcnemar-result">
              <h4 className="mcnemar-label">P-Value:</h4>
              <div className="pvalue-box">
                <strong>{test.p_value.toExponential(6)}</strong>
              </div>

              <h4>Interpretation:</h4>
              <div className="interpretation">
                <p>
                  {test.decision === 'reject_null' ? 'Reject the null hypothesis: paired classification error rates differ.' : 'Fail to reject the null hypothesis: insufficient evidence of different paired error rates.'}
                </p>
                <p>{test.status}. Discordant pairs: {test.discordant_pairs}. Statistic: {test.statistic ?? 'N/A'}.</p>
                {test.supplementary && <p>Small-discordance caution. Supplementary exact p-value: {test.supplementary.p_value.toPrecision(6)}.</p>}
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer className="results-actions">
        {error && <p role="alert">{error}</p>}
        <button type="button" className="results-action download-button" disabled={downloading} onClick={download}>{downloading ? 'Preparing download...' : 'Download Results'}</button>
        <Link className="results-action analyze-button" to="/researcher-upload">Analyze Another CSV</Link>
      </footer>
      {activeInfoTopic && <InfoModal topic={activeInfoTopic} onClose={closeInfo} />}
      </main>
    </div>
  )
}

export default ResearcherResultsPage
