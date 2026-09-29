import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import "./TransactionTable.css"
import { api, label } from "../researchApi"

const Prediction = ({ value }) => (
  <span className="prediction">
    <span className={`statusDot ${value === "Fraudulent" ? "red" : "green"}`} aria-hidden="true" />
    {value}
  </span>
)

const PageJump = ({ currentPage, totalPages, onPageChange }) => {
  const [pageInput, setPageInput] = useState(String(currentPage))

  const goToPage = (event) => {
    event.preventDefault()
    const pageText = pageInput.trim()
    const page = Number(pageText)
    if (!/^\d+$/.test(pageText) || !Number.isInteger(page) || page < 1 || page > totalPages) {
      setPageInput(String(currentPage))
      return
    }

    setPageInput(String(page))
    onPageChange(page)
  }

  return (
    <form className="pageJump" onSubmit={goToPage} noValidate>
      <label htmlFor="transaction-page-input">Page</label>
      <input
        id="transaction-page-input"
        type="text"
        inputMode="numeric"
        enterKeyHint="go"
        autoComplete="off"
        value={pageInput}
        style={{ width: `max(46px, calc(${pageInput.length}ch + 14px))` }}
        aria-describedby="page-jump-help"
        onChange={(event) => setPageInput(event.target.value)}
      />
      <span>of {totalPages}</span>
      <span id="page-jump-help" className="visually-hidden">Press Enter to go to this page.</span>
    </form>
  )
}

const TransactionTable = ({ analysisId, returnQuery, searchTerm, model, outcome, currentPage, onPageChange }) => {
  const navigate = useNavigate()
  const rowsPerPage = 10
  const showSmote = model !== "benchmark"
  const showBenchmark = model !== "rfSmote"
  const [data, setData] = useState({ records: [], total: 0, page: 1 })
  const [error, setError] = useState('')
  useEffect(() => {
    let live = true
    let timer
    const requested = new Set()
    const poll = async () => {
      try {
        const query = new URLSearchParams({ search: searchTerm, search_field: 'row_number', model: model === 'rfSmote' ? 'rf_smote' : model === 'benchmark' ? 'rf' : 'both', outcome, page: currentPage })
        const result = await api(`/analyses/${analysisId}/records?${query}`)
        if (!live) return
        setData(result)
        setError('')
        for (const row of result.records) {
          if (!live) break
          if (row.explanation.status === 'not_requested' && !requested.has(row.row_number)) {
            requested.add(row.row_number)
            try { await api(`/analyses/${analysisId}/records/${row.row_number}/explanation`, { method: 'POST' }) }
            catch (err) { requested.delete(row.row_number); if (live) setError(err.message) }
          }
        }
        if (live && result.records.some(r => ['not_requested', 'pending'].includes(r.explanation.status))) timer = setTimeout(poll, 1500)
      } catch (err) { if (live) setError(err.message) }
    }
    poll()
    return () => { live = false; clearTimeout(timer) }
  }, [analysisId, searchTerm, model, outcome, currentPage])
  const feature = (row, model) => {
    const top = row.explanation.models?.[model]?.top_positive_contributor
    return top ? top.feature || 'No positive contributor' : row.explanation.status === 'failed' ? 'Failed - open details to retry' : 'Pending...'
  }
  const currentRows = data.records.map(row => ({ id: row.transaction_id, number: row.row_number,
    rfSmote: label(row.rf_smote.predicted_label), benchmark: label(row.rf.predicted_label), groundTruth: label(row.actual_label),
    risk1: `${(row.rf.score * 100).toFixed(2)}%`, risk2: `${(row.rf_smote.score * 100).toFixed(2)}%`,
    smoteFeature: feature(row, 'rf_smote'), rfFeature: feature(row, 'rf') }))
  const totalPages = Math.max(1, Math.ceil(data.total / rowsPerPage))
  const startIndex = (data.page - 1) * rowsPerPage
  const visibleColumns = 5 + Number(showSmote) + Number(showBenchmark)
  const rangeStart = data.total ? startIndex + 1 : 0
  const rangeEnd = Math.min(startIndex + rowsPerPage, data.total)

  return (
    <div className="tableContainer">
      {error && <p role="alert">{error}</p>}
      <div className="tableWrapper" role="region" aria-label="Transaction records; scroll horizontally for more columns" tabIndex="0">
        <table className="transactionTable">
          <caption className="visually-hidden">Transaction records with each model's prediction, risk score, and top risk-increasing contributor</caption>
          <thead>
            <tr>
              <th scope="col" className="idColumn">Transaction ID</th>
              {showSmote && <th scope="col" className="smoteText">RF-SMOTE</th>}
              {showBenchmark && <th scope="col" className="benchmarkText">Benchmark RF</th>}
              <th scope="col">Ground Truth</th>
              <th scope="col">Risk Score</th>
              <th scope="col">Top Risk-Increasing Contributor</th>
              <th scope="col">Action</th>
            </tr>
          </thead>
          <tbody>
            {currentRows.map((transaction) => (
              <tr key={transaction.id}>
                <th scope="row" className="idColumn">{transaction.number}</th>
                {showSmote && <td><Prediction value={transaction.rfSmote} /></td>}
                {showBenchmark && <td><Prediction value={transaction.benchmark} /></td>}
                <td><Prediction value={transaction.groundTruth} /></td>
                <td>
                  <div className="badgePair">
                    {showSmote && <span className="riskBadge blueBadge" aria-label={`RF-SMOTE risk score ${transaction.risk2}`}>{transaction.risk2}</span>}
                    {showBenchmark && <span className="riskBadge darkBadge" aria-label={`Benchmark RF risk score ${transaction.risk1}`}>{transaction.risk1}</span>}
                  </div>
                </td>
                <td>
                  <div className="badgePair">
                    {showSmote && <span className="featureBadge blueFeature" aria-label={`RF-SMOTE top contributor ${transaction.smoteFeature}`}>{transaction.smoteFeature}</span>}
                    {showBenchmark && <span className="featureBadge darkFeature" aria-label={`Benchmark RF top contributor ${transaction.rfFeature}`}>{transaction.rfFeature}</span>}
                  </div>
                </td>
                <td>
                  <button type="button" className="viewButton" onClick={() => navigate(`/researcher/transaction/${transaction.number}?analysis=${analysisId}&return=${encodeURIComponent(returnQuery)}`)} aria-label={`View transaction ${transaction.number}`}>View</button>
                </td>
              </tr>
            ))}
            {currentRows.length === 0 && (
              <tr><td colSpan={visibleColumns} className="noRecords">No matching transactions found.</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="pagination">
        <span className="paginationRecords" aria-live="polite">
          Showing {rangeStart}&ndash;{rangeEnd} of {data.total} matching Transaction Records
        </span>
        <div className="paginationControls">
          <button type="button" className="paginationButton" aria-label="Previous page" onClick={() => onPageChange(currentPage - 1)} disabled={currentPage === 1}>&#9664;</button>
          <PageJump key={`${currentPage}-${model}-${outcome}-${searchTerm}`} currentPage={currentPage} totalPages={totalPages} onPageChange={onPageChange} />
          <button type="button" className="paginationButton" aria-label="Next page" onClick={() => onPageChange(currentPage + 1)} disabled={currentPage >= totalPages}>&#9654;</button>
        </div>
      </div>
    </div>
  )
}

export default TransactionTable
