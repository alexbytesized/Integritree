import "./TransactionTable.css"

const transactions = [
  {
    id: "1234567",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "03%",
    risk2: "02%",
    topFeature: "is_merchant_dest"
  },
  {
    id: "1573689",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "01%",
    risk2: "05%",
    topFeature: "type_PAYMENT"
  },
  {
    id: "1864690",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "04%",
    risk2: "12%",
    topFeature: "hour_of_day"
  },
  {
    id: "1928471",
    benchmark: "Fraudulent",
    rfSmote: "Fraudulent",
    groundTruth: "Fraudulent",
    risk1: "91%",
    risk2: "96%",
    topFeature: "amount"
  },
  {
    id: "1748392",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "02%",
    risk2: "04%",
    topFeature: "type_PAYMENT"
  },
  {
    id: "1849201",
    benchmark: "Legitimate",
    rfSmote: "Fraudulent",
    groundTruth: "Fraudulent",
    risk1: "42%",
    risk2: "88%",
    topFeature: "type_TRANSFER"
  },
  {
    id: "1458293",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "05%",
    risk2: "07%",
    topFeature: "hour_of_day"
  },
  {
    id: "1983021",
    benchmark: "Fraudulent",
    rfSmote: "Fraudulent",
    groundTruth: "Fraudulent",
    risk1: "89%",
    risk2: "94%",
    topFeature: "log_amount"
  },
  {
    id: "1657382",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "01%",
    risk2: "03%",
    topFeature: "is_merchant_dest"
  },
  {
    id: "1782930",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "06%",
    risk2: "09%",
    topFeature: "type_CASH_OUT"
  },
  {
    id: "1628391",
    benchmark: "Legitimate",
    rfSmote: "Fraudulent",
    groundTruth: "Fraudulent",
    risk1: "39%",
    risk2: "85%",
    topFeature: "amount"
  },
  {
    id: "1938204",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "02%",
    risk2: "01%",
    topFeature: "type_DEBIT"
  },
  {
    id: "1203849",
    benchmark: "Fraudulent",
    rfSmote: "Fraudulent",
    groundTruth: "Fraudulent",
    risk1: "93%",
    risk2: "97%",
    topFeature: "oldbalanceOrg"
  },
  {
    id: "1384920",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "04%",
    risk2: "06%",
    topFeature: "hour_of_day"
  },
  {
    id: "1493028",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "03%",
    risk2: "08%",
    topFeature: "type_PAYMENT"
  },
  {
    id: "1820394",
    benchmark: "Legitimate",
    rfSmote: "Fraudulent",
    groundTruth: "Fraudulent",
    risk1: "48%",
    risk2: "90%",
    topFeature: "type_TRANSFER"
  },
  {
    id: "1293847",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "01%",
    risk2: "02%",
    topFeature: "is_merchant_dest"
  },
  {
    id: "1728394",
    benchmark: "Fraudulent",
    rfSmote: "Fraudulent",
    groundTruth: "Fraudulent",
    risk1: "87%",
    risk2: "92%",
    topFeature: "log_amount"
  },
  {
    id: "1029384",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "05%",
    risk2: "06%",
    topFeature: "type_CASH_IN"
  },
  {
    id: "1920384",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "02%",
    risk2: "04%",
    topFeature: "hour_of_day"
  },
  {
    id: "1839201",
    benchmark: "Fraudulent",
    rfSmote: "Fraudulent",
    groundTruth: "Fraudulent",
    risk1: "95%",
    risk2: "98%",
    topFeature: "amount"
  },
  {
    id: "1729304",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "07%",
    risk2: "09%",
    topFeature: "type_PAYMENT"
  },
  {
    id: "1392038",
    benchmark: "Legitimate",
    rfSmote: "Fraudulent",
    groundTruth: "Fraudulent",
    risk1: "44%",
    risk2: "82%",
    topFeature: "type_TRANSFER"
  },
  {
    id: "1203948",
    benchmark: "Legitimate",
    rfSmote: "Legitimate",
    groundTruth: "Legitimate",
    risk1: "03%",
    risk2: "05%",
    topFeature: "is_merchant_dest"
  },
  {
    id: "1840293",
    benchmark: "Fraudulent",
    rfSmote: "Fraudulent",
    groundTruth: "Fraudulent",
    risk1: "90%",
    risk2: "95%",
    topFeature: "log_amount"
  }
]

const outcomeFor = (prediction, groundTruth) => {
  if (!groundTruth) return null
  if (prediction === "Fraudulent") return groundTruth === "Fraudulent" ? "tp" : "fp"
  return groundTruth === "Legitimate" ? "tn" : "fn"
}

const Prediction = ({ value }) => (
  <span className="prediction">
    <span className={`statusDot ${value === "Fraudulent" ? "red" : "green"}`} aria-hidden="true" />
    {value}
  </span>
)

const TransactionTable = ({ searchTerm, model, outcome, currentPage, onPageChange }) => {
  const rowsPerPage = 10
  const showSmote = model !== "benchmark"
  const showBenchmark = model !== "rfSmote"
  const normalizedSearch = searchTerm.trim()
  const filteredTransactions = transactions.filter((transaction) => {
    if (normalizedSearch && !transaction.id.includes(normalizedSearch)) return false
    if (outcome !== "all" && model !== "both") {
      const prediction = model === "rfSmote" ? transaction.rfSmote : transaction.benchmark
      return outcomeFor(prediction, transaction.groundTruth) === outcome
    }
    return true
  })
  const totalPages = Math.max(1, Math.ceil(filteredTransactions.length / rowsPerPage))
  const startIndex = (currentPage - 1) * rowsPerPage
  const currentRows = filteredTransactions.slice(startIndex, startIndex + rowsPerPage)
  const visibleColumns = 5 + Number(showSmote) + Number(showBenchmark)
  const rangeStart = filteredTransactions.length ? startIndex + 1 : 0
  const rangeEnd = Math.min(startIndex + rowsPerPage, filteredTransactions.length)

  return (
    <div className="tableContainer">
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
                <th scope="row" className="idColumn">{transaction.id}</th>
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
                    {showSmote && <span className="featureBadge blueFeature" aria-label={`RF-SMOTE top contributor ${transaction.topFeature}`}>{transaction.topFeature}</span>}
                    {showBenchmark && <span className="featureBadge darkFeature" aria-label={`Benchmark RF top contributor ${transaction.topFeature}`}>{transaction.topFeature}</span>}
                  </div>
                </td>
                <td>
                  <button type="button" className="viewButton" aria-disabled="true" title="Transaction details are not available yet">View<span className="visually-hidden"> transaction {transaction.id}; details are not available yet</span></button>
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
          Showing {rangeStart}&ndash;{rangeEnd} of {filteredTransactions.length} matching Transaction Records ({transactions.length} total)
        </span>
        <div className="paginationControls">
          <button type="button" className="paginationButton" aria-label="Previous page" onClick={() => onPageChange(currentPage - 1)} disabled={currentPage === 1}>&#9664;</button>
          <span className="paginationText">Page {currentPage} of {totalPages}</span>
          <button type="button" className="paginationButton" aria-label="Next page" onClick={() => onPageChange(currentPage + 1)} disabled={currentPage === totalPages}>&#9654;</button>
        </div>
      </div>
    </div>
  )
}

export default TransactionTable
