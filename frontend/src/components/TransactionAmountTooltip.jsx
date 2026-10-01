import { useEffect, useId, useRef, useState } from 'react'
import { Info } from 'lucide-react'
import './RiskThresholdTooltip.css'

export default function TransactionAmountTooltip() {
  const [open, setOpen] = useState(false)
  const root = useRef(null)
  const id = useId()
  useEffect(() => {
    const outside = event => { if (!root.current?.contains(event.target)) setOpen(false) }
    const escape = event => { if (event.key === 'Escape') setOpen(false) }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('keydown', escape)
    }
  }, [])
  return <div className="risk-tooltip-wrapper receipt-amount-help" ref={root}
    onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
    onFocus={() => setOpen(true)} onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false) }}>
    <button type="button" className="risk-tooltip-trigger" aria-label="Transaction Amount Details"
      aria-expanded={open} aria-describedby={open ? id : undefined} onClick={() => setOpen(true)}><Info size={16} /></button>
    {open && <div id={id} role="tooltip" className="risk-tooltip-popover risk-tooltip--bottom">
      <div className="risk-tooltip-header"><h4 className="risk-tooltip-title">Transaction Amount Details</h4></div>
      <p>Transaction amount refers to the PHP principal amount (excluding fees)</p>
    </div>}
  </div>
}
