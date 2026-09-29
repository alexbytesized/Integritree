import { Link } from 'react-router-dom'
import './ResearchAnalysisLoading.css'

const STATUS_MESSAGES = {
  uploading: 'Uploading your CSV…',
  queued: 'Waiting to start your analysis…',
  loading_models: 'Verifying fraud detection models…',
  processing: 'Analyzing your transaction records…',
  evaluating: 'Calculating your analysis results…',
}

export default function ResearchAnalysisLoading({ status, rowsProcessed = 0 }) {
  const showCount = status === 'processing' || status === 'evaluating'
  return <main className="research-loading">
    <h1 className="visually-hidden">CSV analysis</h1>
    <div className="researcher-results-page research-loading-skeleton" aria-hidden="true">
      <div className="research-loading-placeholder research-loading-heading" />
      <div className="research-loading-banner">
        <div className="research-loading-placeholder" />
      </div>
      <div className="research-loading-summaries">
        {[0, 1, 2].map(index => <div className="research-loading-summary" key={index}>
          <div className="research-loading-placeholder research-loading-card-title" />
          <div className="research-loading-donut" />
          <div className="research-loading-placeholder research-loading-card-value" />
          <div className="research-loading-placeholder research-loading-card-caption" />
        </div>)}
      </div>
      <div className="research-loading-placeholder research-loading-table-heading" />
      <div className="research-loading-filters">
        {[0, 1, 2].map(index => <div className="research-loading-placeholder" key={index} />)}
      </div>
      <div className="research-loading-table">
        <div className="research-loading-table-header">
          {[0, 1, 2, 3, 4, 5].map(index => <div className="research-loading-placeholder" key={index} />)}
        </div>
        {Array.from({ length: 6 }, (_, row) => <div className="research-loading-table-row" key={row}>
          {[0, 1, 2, 3, 4, 5].map(column => <div className="research-loading-placeholder" key={column} />)}
        </div>)}
      </div>
    </div>
    <div className="research-loading-overlay">
      <div className="research-loading-status">
        <div className="research-loading-spinner" aria-hidden="true" />
        <p className="research-loading-message" role="status" aria-live="polite" aria-atomic="true">
          {STATUS_MESSAGES[status] || 'Connecting to your analysis…'}
        </p>
        {showCount && <p className="research-loading-count">{rowsProcessed.toLocaleString('en-US')} records processed.</p>}
        <Link className="research-loading-return" to="/researcher-upload">Return to upload</Link>
      </div>
    </div>
  </main>
}
