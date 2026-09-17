import React from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import './Navbar.css'

const Navbar = () => {
  const navigate = useNavigate()
  const location = useLocation()

  const handleNavClick = (e, id) => {
    e.preventDefault()
    if (location.pathname !== '/') {
      navigate('/', { state: { scrollToId: id } })
    } else {
      const element = document.getElementById(id)
      if (element) {
        const y = element.getBoundingClientRect().top + window.pageYOffset
        window.scrollTo({ top: y, behavior: 'smooth' })
      }
    }
  }

  return (
    <div className="navbar-container">
      <nav className="navbar">
        {/* Logo */}
        <Link to="/" className="navbar-logo">
          INTEGRITREE
        </Link>

        {/* Nav Links */}
        <div className="navbar-links">
          <a href="#features" onClick={(e) => handleNavClick(e, 'features-anchor')} className="nav-link">Features</a>
          <a href="#how-it-works" onClick={(e) => handleNavClick(e, 'how-it-works-anchor')} className="nav-link">How It Works</a>
          <a href="#meet-the-team" onClick={(e) => handleNavClick(e, 'meet-the-team-anchor')} className="nav-link">Meet the Team</a>
        </div>

        {/* CTA Button */}
        <div className="navbar-actions">
          <Link to="/upload" className="btn-begin">
            <span className="btn-begin-icon">▶</span> Begin
          </Link>
        </div>
      </nav>
    </div>
  )
}

export default Navbar