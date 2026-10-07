import { ArrowLeft } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import './ReturnButton.css'

const ReturnButton = ({ to, onClick, disabled = false }) => {
  const navigate = useNavigate()

  if (onClick) {
    return (
      <button type="button" onClick={onClick} disabled={disabled} className="return-btn">
        <ArrowLeft size={18} strokeWidth={2.5} />
        Return
      </button>
    )
  }

  if (to === -1) {
    return (
      <button type="button" onClick={() => navigate(-1)} className="return-btn">
        <ArrowLeft size={18} strokeWidth={2.5} />
        Return
      </button>
    )
  }

  return (
    <Link to={to || "/"} className="return-btn">
      <ArrowLeft size={18} strokeWidth={2.5} />
      Return
    </Link>
  )
}

export default ReturnButton
