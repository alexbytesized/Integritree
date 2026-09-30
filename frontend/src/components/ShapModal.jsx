import { useEffect, useId, useRef } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import { MODEL_NAMES } from '../researchDisplay'
import ResearchExplanation from './ResearchExplanation'
import './InfoModal.css'
import './ShapModal.css'

export default function ShapModal({ model, explanation, onRetry, onClose, chartApi, chartPresentation }) {
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
    close.current.focus()
    return () => {
      document.body.style.overflow = previousOverflow
      if (background) background.inert = wasInert
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [])
  const handleKey = event => {
    if (event.key === 'Escape') { event.preventDefault(); onClose() }
    if (event.key === 'Tab') {
      const targets = [...dialog.current.querySelectorAll('button:not([disabled]), a[href], [tabindex="0"]')]
      const first = targets[0], last = targets[targets.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
  }
  return createPortal(<div className="info-modal-overlay" onClick={event => { if (event.target === event.currentTarget) onClose() }}>
    <div ref={dialog} className="info-modal-dialog shap-modal" role="dialog" aria-modal="true" aria-labelledby={titleId} onKeyDown={handleKey}>
      <div className="info-modal-title-bar"><h2 id={titleId}>{MODEL_NAMES[model]} — SHAP Evaluation</h2></div>
      <div className="info-modal-panel">
        <button ref={close} type="button" className="info-modal-close" aria-label="Close SHAP evaluation" onClick={onClose}><X aria-hidden="true" /></button>
        <div className="info-modal-scroll" tabIndex="0" role="region" aria-label="SHAP evaluation contents">
          <ResearchExplanation explanation={explanation} model={model} onRetry={onRetry} chartApi={chartApi} chartPresentation={chartPresentation} />
        </div>
      </div>
    </div>
  </div>, document.body)
}
