import { useEffect, useId, useRef } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import './InfoModal.css'

export default function UploadInstructionsModal({ onClose }) {
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
          <h2 id={titleId}>Upload Instructions</h2>
        </div>
        <div className="info-modal-panel">
          <button
            ref={close}
            type="button"
            className="info-modal-close"
            aria-label="Close upload instructions"
            onClick={onClose}
          >
            <X aria-hidden="true" />
          </button>
          <div className="info-modal-scroll" tabIndex="0" role="region" aria-label="Upload instructions content">
            <div className="info-modal-section">
              <h3>Supported Screenshots</h3>
              <p>
                Upload GCash Express Send, Pay Online, or bank-account transfer screenshots. You can edit all five transaction types after extraction; Cash In/Out and QR screenshot layouts are not supported yet.
              </p>
            </div>
            <div className="info-modal-section">
              <h3>Review &amp; Confirmation</h3>
              <p>
                Review and confirm extracted details before prediction. Receipts clear automatically when you leave the receipt flow or close this tab. Refresh preserves your current receipt.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>,
    document.body
  )
}
