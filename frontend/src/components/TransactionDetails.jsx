import { useState } from "react"

import "./TransactionDetails.css"

const TransactionDetails = ({ initialData = {}, onChange }) => {
  const [transaction, setTransaction] = useState({
    transactionId: initialData.transactionId ?? "12345678",
    amount: initialData.amount ?? "₱5,000.00",
    transactionType: initialData.transactionType ?? "Transfer",
    transactionDate: initialData.transactionDate ?? "September 14, 2026",
    transactionTime: initialData.transactionTime ?? "3:15 AM",
    senderType: initialData.senderType ?? "Customer",
    recipient: initialData.recipient ?? "ABC Store"
  })

  const handleChange = (event) => {
    const { name, value } = event.target

    const updatedTransaction = {
      ...transaction,
      [name]: value
    }

    setTransaction(updatedTransaction)

    if (onChange) {
      onChange(updatedTransaction)
    }
  }

  return (
    <section className="transaction-details-container">
      <div className="transaction-title">
        <h2>Transaction Details</h2>
      </div>

      <div className="transaction-details-input-container">
        <div className="details-row">
          <label htmlFor="transaction-id">Transaction ID:</label>

          <input
            id="transaction-id"
            name="transactionId"
            type="text"
            value={transaction.transactionId}
            onChange={handleChange}
          />
        </div>

        <div className="details-row">
          <label htmlFor="transaction-amount">Transaction Amount:</label>

          <input id="transaction-amount" name="amount" type="text" value={transaction.amount} onChange={handleChange} />
        </div>

        <div className="details-row">
          <label htmlFor="transaction-type">Transaction Type:</label>

          <select
            id="transaction-type"
            name="transactionType"
            value={transaction.transactionType}
            onChange={handleChange}
          >
            <option value="Transfer">Transfer</option>
            <option value="Payment">Payment</option>
            <option value="Cash In">Cash In</option>
            <option value="Cash Out">Cash Out</option>
            <option value="Debit">Debit</option>
          </select>
        </div>

        <div className="details-row">
          <label htmlFor="transaction-date">Transaction Date:</label>

          <input
            id="transaction-date"
            name="transactionDate"
            type="text"
            value={transaction.transactionDate}
            onChange={handleChange}
          />
        </div>

        <div className="details-row">
          <label htmlFor="transaction-time">Transaction Time:</label>

          <input
            id="transaction-time"
            name="transactionTime"
            type="text"
            value={transaction.transactionTime}
            onChange={handleChange}
          />
        </div>

        <div className="details-row">
          <label htmlFor="sender-type">Sender Type:</label>

          <select id="sender-type" name="senderType" value={transaction.senderType} onChange={handleChange}>
            <option value="Customer">Customer</option>
            <option value="Merchant">Merchant</option>
          </select>
        </div>

        <div className="details-row">
          <label htmlFor="recipient">Recipient:</label>

          <input id="recipient" name="recipient" type="text" value={transaction.recipient} onChange={handleChange} />
        </div>
      </div>
    </section>
  )
}

export default TransactionDetails
