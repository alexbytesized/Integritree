import { ArrowLeft } from 'lucide-react'
import { Link } from 'react-router-dom'
import './ReturnButton.css'

const ReturnButton = ({ to = '/' }) => (
  <Link to={to} className="return-btn">
    <ArrowLeft size={18} strokeWidth={2.5} />
    Return
  </Link>
)

export default ReturnButton
