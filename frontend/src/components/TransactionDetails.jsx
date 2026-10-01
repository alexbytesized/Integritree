import { TRANSACTION_TYPES } from '../receiptDisplay'
import './TransactionDetails.css'
import TransactionAmountTooltip from './TransactionAmountTooltip'

export default function TransactionDetails({ value, onChange, disabled = false }) {
  const input = (name, props = {}) => ({ id: `receipt-${name}`, name, value: value[name],
    disabled, onChange: event => {
      const next = event.target.value
      if (name !== 'amount' || /^[0-9]*([.][0-9]{0,2})?$/.test(next)) onChange(name, next)
    }, ...props })
  const row = (name, label, control) => <div className="details-row" key={name}>
    <label htmlFor={`receipt-${name}`}>{label}</label>{control}
  </div>
  const roles = name => <select {...input(name)} required>
    <option value="client">Client</option><option value="merchant">Merchant</option>
  </select>
  return <section className="transaction-details-container">
    <div className="transaction-title"><h2>Transaction Details</h2></div>
    <div className="transaction-details-input-container">
      {row('reference', 'Transaction ID:', <input {...input('reference')} maxLength={256} title="Receipt reference (optional)" />)}
      {row('amount', 'Transaction Amount:', <div className="receipt-amount-control"><input {...input('amount')} inputMode="decimal" required maxLength={32} pattern="[0-9]+([.][0-9]{1,2})?" /><TransactionAmountTooltip /></div>)}
      {row('category', 'Transaction Type:', <select {...input('category')} required>
        {Object.entries(TRANSACTION_TYPES).map(([key, type]) => <option key={key} value={key}>{type.label}</option>)}
      </select>)}
      {row('date', 'Transaction Date:', <input {...input('date')} type="date" required />)}
      {row('time', 'Transaction Time:', <input {...input('time')} type="time" step="60" required />)}
      {row('origin_role', 'Sender Type:', roles('origin_role'))}
      {row('destination_role', 'Recipient Type:', roles('destination_role'))}
    </div>
  </section>
}
