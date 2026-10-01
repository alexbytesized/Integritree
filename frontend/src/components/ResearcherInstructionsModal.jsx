import { useEffect, useId, useRef } from 'react'
import { createPortal } from 'react-dom'
import { X, Download } from 'lucide-react'
import './InfoModal.css'

export default function ResearcherInstructionsModal({ onClose }) {
  const dialog = useRef(null)
  const close = useRef(null)
  const titleId = useId()

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
      const targets = [...dialog.current.querySelectorAll('button:not([disabled]), a[href], [tabindex="0"]')]
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
        className="info-modal-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onKeyDown={handleKey}
      >
        <div className="info-modal-title-bar">
          <h2 id={titleId}>Dataset &amp; Upload Guidelines</h2>
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
              <h3>Required Columns &amp; Format</h3>
              <p>
                Required columns: <strong>step</strong>, <strong>type</strong>, <strong>amount</strong>, <strong>nameOrig</strong>, <strong>nameDest</strong>, <strong>isFraud</strong> (0 or 1). File must be a UTF-8 encoded CSV with no duplicate headers.
              </p>
            </div>
            <div className="info-modal-section">
              <h3>Validation &amp; Column Rules</h3>
              <p>
                <strong>step</strong> must be a positive integer; account names must start with <strong>C</strong> or <strong>M</strong>. Allowed transaction types: <strong>CASH_IN</strong>, <strong>CASH_OUT</strong>, <strong>DEBIT</strong>, <strong>PAYMENT</strong>, <strong>TRANSFER</strong>, or <strong>unknown</strong>. Missing amount or type values will use the saved preprocessing rules. Other original PaySim columns are optional and are not predictors.
              </p>
            </div>
            <div className="info-modal-section">
              <h3>Demonstration Template</h3>
              <p>
                Download a pre-formatted sample dataset to test the analysis workflow:
              </p>
              <div>
                <a href="/api/v1/research/template" download className="instructions-download-btn">
                  <Download size={16} />
                  Download Demonstration CSV Template
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
