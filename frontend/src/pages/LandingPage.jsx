import React, { useEffect } from 'react' // 1. Added useEffect
import { Link } from 'react-router-dom'
import Navbar from '../components/Navbar'
import treeSvg from '../assets/tree.svg'
import './LandingPage.css'

const LandingPage = () => {

  // Initialize particles effect
  useEffect(() => {
    if (window.particlesJS) {
      window.particlesJS.load('particles-js', '/particles.json', function() {
        console.log('callback - particles.js config loaded');
      });
    }
  }, []);

  return (
    <div className="landing-page-container">
      <div id="particles-js"></div> 
      
      <Navbar />
      
      <main className="hero-section">
        <div className="hero-left">
          <h1 className="hero-title">
            INTEGRI<span className="hero-title-accent">TREE</span>
          </h1>
          <div className="hero-tagline-row">
            <p className="hero-tagline">
              Branching Toward Integrity,<br/>
              Powered by Algorithmic Trees.
            </p>
            <Link to="/upload" className="hero-btn-begin">
              <span className="hero-btn-begin-icon">▶</span> BEGIN
            </Link>
          </div>
        </div>
        
        <div className="hero-right">
          <div className="hero-tree-container">
            <img src={treeSvg} alt="Integritree Wireframe Tree" className="hero-tree-svg" />
          </div>
        </div>
      </main>
    </div>
  )
}

export default LandingPage