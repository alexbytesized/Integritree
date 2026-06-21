import React from 'react'
import { Link } from 'react-router-dom'
import './Navbar.css'

const Navbar = () => {
  return (
    <div className="navbar-container">
      <nav className="navbar">
        <Link to="/" className="navbar-logo">INTEGRITREE</Link>
        <div className="navbar-links">
          <a href="#features" className="nav-link">Features</a>
          <a href="#how-it-works" className="nav-link">How It Works</a>
          <a href="#meet-the-team" className="nav-link">Meet the Team</a>
        </div>
        <div className="navbar-actions">
          <Link to="/upload" className="btn-begin">
            <span className="btn-begin-icon">▶</span> BEGIN
          </Link>
        </div>
      </nav>
    </div>
  )
}

export default Navbar