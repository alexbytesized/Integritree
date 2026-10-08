import { useEffect, useId, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { X, Download } from 'lucide-react'
import './InfoModal.css'
import './ResearcherInstructionsModal.css'

const formats = [
  { key: 'raw', label: 'Raw CSV', columns: ['step', 'type', 'amount', 'nameOrig', 'nameDest', 'isFraud'] },
  { key: 'prepared', label: 'Preprocessed CSV', columns: [
    'hour_of_day', 'day_of_week', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
    'type_PAYMENT', 'type_TRANSFER', 'log_amount', 'is_zero_amount',
    'is_merchant_origin', 'is_merchant_dest', 'isFraud',
  ] },
]

export default function ResearcherInstructionsModal({ onClose }) {
  const dialog = useRef(null)
  const close = useRef(null)
  const titleId = useId()
  const [activeFormat, setActiveFormat] = useState('raw')
  const tabs = useRef([])

  const handleTabKey = (event, index) => {
    const next = { ArrowRight: (index + 1) % formats.length,
      ArrowLeft: (index + formats.length - 1) % formats.length,
      Home: 0, End: formats.length - 1 }[event.key]
    if (next === undefined) return
    event.preventDefault()
    setActiveFormat(formats[next].key)
    tabs.current[next].focus()
  }

  useEffect(() => {
    const previousFocus = document.activeElement
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const background = document.getElementById('root')
    const wasInert = background?.inert
    if (background) background.inert = true
    close.current?.focus()
    return () => {
      document.body.style.overflow = previousOverflow
      if (background) background.inert = wasInert
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [])

  const handleKey = event => {
    if (event.key === 'Escape') {
      event.preventDefault()
      onClose()
    }
    if (event.key === 'Tab') {
      const targets = [...dialog.current.querySelectorAll('button:not([disabled]), a[href], summary, [tabindex="0"]')]
        .filter(element => element.tabIndex >= 0 && element.getClientRects().length > 0)
      const first = targets[0]
      const last = targets[targets.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
  }

  return createPortal(
    <div
      className="info-modal-overlay"
      onClick={event => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div
        ref={dialog}
        className="info-modal-dialog researcher-guide"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onKeyDown={handleKey}
      >
        <div className="info-modal-title-bar">
          <h2 id={titleId}>CSV Upload Guide</h2>
        </div>
        <div className="info-modal-panel">
          <button
            ref={close}
            type="button"
            className="info-modal-close"
            aria-label="Close dataset instructions"
            onClick={onClose}
          >
            <X aria-hidden="true" />
          </button>
          <div className="info-modal-scroll" tabIndex="0" role="region" aria-label="Dataset instructions content">
            <div className="info-modal-section">
              <h3>Find your file format</h3>
              <div className="researcher-guide-tabs" role="tablist" aria-label="CSV format instructions">
                {formats.map((format, index) => (
                  <button key={format.key} ref={element => { tabs.current[index] = element }}
                    type="button" role="tab" id={`${titleId}-${format.key}-tab`}
                    aria-controls={`${titleId}-${format.key}-panel`}
                    aria-selected={activeFormat === format.key}
                    tabIndex={activeFormat === format.key ? 0 : -1}
                    onClick={() => setActiveFormat(format.key)}
                    onKeyDown={event => handleTabKey(event, index)}>
                    {format.label}
                  </button>
                ))}
              </div>
              {formats.map(format => (
                <div key={format.key} role="tabpanel" id={`${titleId}-${format.key}-panel`}
                  aria-labelledby={`${titleId}-${format.key}-tab`} hidden={activeFormat !== format.key}>
                  <p>{format.key === 'raw'
                    ? 'Use this format when your file contains the original transaction fields. The system applies the models’ saved preprocessing automatically.'
                    : 'Use this format when your file already contains the complete model features. The system uses those values directly without repeating preprocessing.'}</p>
                  <p className="researcher-guide-column-label"><strong>Required columns:</strong></p>
                  <ul className="researcher-guide-columns" aria-label={`${format.label} required columns`}>
                    {format.columns.map(column => <li key={column}><code>{column}</code></li>)}
                  </ul>
                  {format.key === 'prepared' && <p className="researcher-guide-compatibility">
                    Use features prepared with the same transformations and scaling parameters as the selected models. The system checks the columns and values, but cannot verify how the file was preprocessed.
                  </p>}
                  <details className="researcher-guide-values">
                    <summary>Value requirements</summary>
                    <ul>
                      {format.key === 'raw' ? <>
                        <li><code>step</code> must be a positive integer.</li>
                        <li><code>type</code> must be CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER, or unknown. A blank type is treated as unknown.</li>
                        <li><code>amount</code> must be a finite, nonnegative number. A blank amount is filled using the saved training median.</li>
                        <li><code>nameOrig</code> and <code>nameDest</code> must begin with C or M, followed by an identifier.</li>
                      </> : <>
                        <li>All feature values must be finite numbers with no missing entries.</li>
                        <li>Binary indicators must be 0 or 1.</li>
                        <li>At most one transaction-type indicator may be 1. All-zero type indicators mean unknown.</li>
                        <li>Scaled values may fall outside [0, 1].</li>
                      </>}
                      <li><code>isFraud</code> must be 0 for legitimate or 1 for fraudulent.</li>
                    </ul>
                  </details>
                </div>
              ))}
            </div>
            <div className="info-modal-section">
              <h3>File requirements</h3>
              <ul>
                <li>Save as a <strong>UTF-8 CSV</strong>, no larger than <strong>500 MiB</strong>.</li>
                <li>Use the exact column names and capitalization. Column order may vary.</li>
                <li>Do not duplicate column names or mix raw and preprocessed columns.</li>
                <li>Keep each transaction’s <code>isFraud</code> label aligned with its feature values.</li>
              </ul>
            </div>
            <div className="info-modal-section">
              <h3>Need an example?</h3>
              <p>
                This example demonstrates the required file structure. It is not the study’s validation or test dataset.
              </p>
              <div>
                <a href="/api/v1/research/template" download className="instructions-download-btn">
                  <Download size={16} />
                  Download Raw CSV Example
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>,
    document.body
  )
}
