import ReceiptParticles from './ReceiptParticles'
import ReturnButton from './ReturnButton'
import stars from '../assets/stars.png'
import './UploadErrorPage.css'

export default function UploadErrorPage({ message, details, issues, clearLabel, onClear,
  busy = false, actionError, returnTo, onReturn, onRetry }) {
  const rowIssues = Object.entries(issues || {})
  return <div className="upload-error-wrapper">
    <ReceiptParticles />
    <ReturnButton to={returnTo} onClick={onReturn} disabled={busy} />
    <main className="upload-error-page">
      <div className="upload-error-heading">
        <h1>Error Encountered</h1><img src={stars} alt="" />
      </div>
      <p role="alert">{message}</p>
      {(details || rowIssues.length > 0) && <details className="upload-error-details">
        <summary>View details</summary>
        {details && <p>{details}</p>}
        {rowIssues.length > 0 && <>
          <p>Row numbers count data records, excluding the header.</p>
          <ul>{rowIssues.map(([field, issue]) => <li key={field}>
            <strong>{field}</strong>: {issue.count != null && `${issue.count} invalid value(s). `}
            {issue.rows?.length > 0 && `Affected rows${issue.count > issue.rows.length ? ' (first reported)' : ''}: ${issue.rows.join(', ')}.`}
          </li>)}</ul>
        </>}
      </details>}
      {actionError && <p role="alert">{actionError}</p>}
      <div className="upload-error-actions">
        {onRetry && <button type="button" disabled={busy} onClick={onRetry}>Retry processing</button>}
        {onClear && <button type="button" disabled={busy} onClick={onClear}>{clearLabel}</button>}
      </div>
    </main>
  </div>
}
