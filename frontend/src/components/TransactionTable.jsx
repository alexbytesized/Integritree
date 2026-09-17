import { useEffect, useState } from "react"
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

const TransactionTable = ({ searchTerm }) => {
  const [currentPage, setCurrentPage] = useState(1)

  const rowsPerPage = 10

  const normalizedSearch = searchTerm.trim()

  const filteredTransactions = transactions.filter((transaction) => {
    if (normalizedSearch === "") return true

    return transaction.id.startsWith(normalizedSearch)
  })

  const totalPages = Math.ceil(filteredTransactions.length / rowsPerPage) || 1

  const startIndex = (currentPage - 1) * rowsPerPage
  const endIndex = startIndex + rowsPerPage

  const currentRows = filteredTransactions.slice(startIndex, endIndex)
  const emptyRows = rowsPerPage - currentRows.length

  useEffect(() => {
    setCurrentPage(1)
  }, [searchTerm])

  const goToPreviousPage = () => {
    setCurrentPage((prevPage) => Math.max(prevPage - 1, 1))
  }

  const goToNextPage = () => {
    setCurrentPage((prevPage) => Math.min(prevPage + 1, totalPages))
  }

  return (
    <div className="tableContainer">
      <div className="tableWrapper">
        <table className="transactionTable">
          <thead>
            <tr>
              <th>Transaction ID</th>
              <th className="blueText">Benchmark RF</th>
              <th className="purpleText">RF-SMOTE</th>
              <th>Ground Truth</th>
              <th>Risk Score</th>
              <th>Top Feature</th>
            </tr>
          </thead>

          <tbody>
            {currentRows.length > 0 ? (
              <>
                {currentRows.map((transaction) => (
                  <tr key={transaction.id}>
                    <td>{transaction.id}</td>

                    <td>
                      <span className={`statusDot ${transaction.benchmark === "Fraudulent" ? "red" : "green"}`} />
                      {transaction.benchmark}
                    </td>

                    <td>
                      <span className={`statusDot ${transaction.rfSmote === "Fraudulent" ? "red" : "green"}`} />
                      {transaction.rfSmote}
                    </td>

                    <td>
                      <span className={`statusDot ${transaction.groundTruth === "Fraudulent" ? "red" : "green"}`} />
                      {transaction.groundTruth}
                    </td>

                    <td>
                      <span className="riskBadge blueBadge">{transaction.risk1}</span>

                      <span className="riskBadge purpleBadge">{transaction.risk2}</span>
                    </td>

                    <td>
                      <span className="featureBadge">{transaction.topFeature}</span>
                    </td>
                  </tr>
                ))}

                {Array.from({ length: emptyRows }).map((_, index) => (
                  <tr className="emptyRow" key={`empty-${index}`}>
                    <td>&nbsp;</td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                  </tr>
                ))}
              </>
            ) : (
              <tr>
                <td colSpan="6" className="noRecords">
                  No matching transactions found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <button className="paginationButton" onClick={goToPreviousPage} disabled={currentPage === 1}>
          ◀
        </button>

        <span className="paginationText">
          Page {currentPage} of {totalPages}
        </span>

        <button className="paginationButton" onClick={goToNextPage} disabled={currentPage === totalPages}>
          ▶
        </button>
      </div>
    </div>
  )
}

export default TransactionTable
