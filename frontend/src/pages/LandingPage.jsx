import React, { useEffect } from 'react' // 1. Added useEffect
import { Link } from 'react-router-dom'
import Navbar from '../components/Navbar'
import treeSvg from '../assets/tree.svg'
import feature1Svg from '../assets/feature1.svg'
import feature2Svg from '../assets/feature2.svg'
import feature3Svg from '../assets/feature3.svg'
import step1Svg from '../assets/step1.svg'
import step2Svg from '../assets/step2.svg'
import step3Svg from '../assets/step3.svg'
import step4Svg from '../assets/step4.svg'
import step5Svg from '../assets/step5.svg'
import step6Svg from '../assets/step6.svg'
import team1Svg from '../assets/team1.svg'
import team2Svg from '../assets/team2.svg'
import team3Svg from '../assets/team3.svg'
import team4Svg from '../assets/team4.svg'
import footerblobsSvg from '../assets/footerblobs.svg'
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
      
      {/* Title Section */}
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
            <Link to="/results" className="hero-btn-begin">
              <span className="hero-btn-begin-icon">▶</span> BEGIN
            </Link>
          </div>
        </div>
        
        <div className="hero-right">
          <div className="hero-tree-container">
            <img src={treeSvg} alt="Integritree Tree" className="hero-tree-svg" />
          </div>
        </div>
      </main>

      {/* Features Section */}
      <section className="features-section">
        <h2 className="features-section-title">
          Enabling Transparent E-Wallet Fraud Detection
        </h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon-container">
              <img src={feature1Svg} alt="Fraud Detection" className="feature-icon" />
            </div>
            <h3 className="feature-title">Fraud Detection</h3>
            <p className="feature-description">
              Distinguish fraudulent and legitimate e-wallet transactions using a trained Random Forest model.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon-container">
              <img src={feature2Svg} alt="Classification Explanation" className="feature-icon" />
            </div>
            <h3 className="feature-title">Classification Explanation</h3>
            <p className="feature-description">
              Provides SHAP-based explanations showing which transaction feature influenced the most for the model's fraud classification.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon-container">
              <img src={feature3Svg} alt="Risk Scoring" className="feature-icon" />
            </div>
            <h3 className="feature-title">Risk Scoring</h3>
            <p className="feature-description">
              Assigns a fraud risk score to each transaction based on the model's predicted fraud probability.
            </p>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section className="how-it-works-section">
        <h2 className="how-it-works-title">Integritree in Action: How it Works</h2>
        <div className="how-it-works-grid">
          {/* Column 1 */}
          <div className="how-it-works-col">
            <div className="how-it-works-item">
              <img src={step1Svg} alt="Step 1" className="step-icon" />
              <span className="step-text">Upload Transactional Data</span>
            </div>
            <div className="how-it-works-item">
              <img src={step2Svg} alt="Step 2" className="step-icon" />
              <span className="step-text">Preprocess the Data</span>
            </div>
            <div className="how-it-works-item">
              <img src={step3Svg} alt="Step 3" className="step-icon" />
              <span className="step-text">Detect Potential Fraud</span>
            </div>
          </div>

          {/* Column 2 */}
          <div className="how-it-works-col">
            <div className="how-it-works-item">
              <img src={step4Svg} alt="Step 4" className="step-icon" />
              <span className="step-text">Generate Risk Scores</span>
            </div>
            <div className="how-it-works-item">
              <img src={step5Svg} alt="Step 5" className="step-icon" />
              <span className="step-text">Explain the Classification</span>
            </div>
            <div className="how-it-works-item">
              <img src={step6Svg} alt="Step 6" className="step-icon" />
              <span className="step-text">Review the Results</span>
            </div>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="team-section">
        <h2 className="team-section-title">Meet the Cultivators Behind Integritree</h2>
        
        <div className="team-grid">
          <div className="team-card">
            <div className="team-member-info">
              <span className="team-member-lastname">Cabrera,</span>
              <span className="team-member-firstname">Chelsea Lauren</span>
            </div>
            <img src={team1Svg} alt="Chelsea Lauren Cabrera" className="team-member-avatar" />
          </div>

          <div className="team-card">
            <div className="team-member-info">
              <span className="team-member-lastname">Carpio,</span>
              <span className="team-member-firstname">Joshua Enzo</span>
            </div>
            <img src={team2Svg} alt="Joshua Enzo Carpio" className="team-member-avatar" />
          </div>

          <div className="team-card">
            <div className="team-member-info">
              <span className="team-member-lastname">Coralde,</span>
              <span className="team-member-firstname">Jule Alexander</span>
            </div>
            <img src={team3Svg} alt="Jule Alexander Coralde" className="team-member-avatar" />
          </div>

          <div className="team-card">
            <div className="team-member-info">
              <span className="team-member-lastname">Matinong,</span>
              <span className="team-member-firstname">Arwen Therese Emmanuelle</span>
            </div>
            <img src={team4Svg} alt="Arwen Therese Emmanuelle Matinong" className="team-member-avatar" />
          </div>
        </div>
      </section>

      {/* Footer Section */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-left">
            <h2 className="footer-title">
              Start Analyzing Transactions
            </h2>
            <p className="footer-info">
              Begin reviewing e-wallet transactions with fraud detection, risk scoring, and classification explanations.
            </p>
          </div>
          
          <img src={footerblobsSvg} className="footer-blobs" />
          
          <div className="footer-right">
            <Link to="/upload" className="footer-btn-begin">
              <span className="footer-btn-begin-icon">▶</span> BEGIN
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default LandingPage