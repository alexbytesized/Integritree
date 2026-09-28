import React from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import './Navbar.css'

const Navbar = () => {
  const navigate = useNavigate()
  const location = useLocation()

  const scrollToTarget = (id) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  const handleNavClick = (e, id) => {
    e.preventDefault()
    if (location.pathname !== '/') {
      navigate('/', { state: { scrollToId: id } })
    } else {
      scrollToTarget(id)
    }
  }

  const handleLogoClick = (e) => {
    if (location.pathname === '/') {
      e.preventDefault()
      window.scrollTo({ top: 0, behavior: 'smooth' })
      if (document.body) {
        document.body.scrollTo({ top: 0, behavior: 'smooth' })
      }
    }
  }

  return (
    <div className="navbar-container">
      <nav className="navbar">
        {/* Logo */}
        <Link to="/" onClick={handleLogoClick} className="navbar-logo">
          INTEGRITREE
        </Link>

        {/* Nav Links */}
        <div className="navbar-links">
          <a
            href="#features-anchor"
            onClick={(e) => handleNavClick(e, 'features-anchor')}
            className="nav-link"
          >
            Features
          </a>
          <a
            href="#how-it-works-anchor"
            onClick={(e) => handleNavClick(e, 'how-it-works-anchor')}
            className="nav-link"
          >
            How It Works
          </a>
          <a
            href="#meet-the-team-anchor"
            onClick={(e) => handleNavClick(e, 'meet-the-team-anchor')}
            className="nav-link"
          >
            Meet the Team
          </a>
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