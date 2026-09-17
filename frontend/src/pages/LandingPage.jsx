import React, { useEffect } from 'react'
import { Link } from 'react-router-dom'
import Navbar from '../components/Navbar'
import starsSvg from '../assets/landingpage-stars.svg'
import feature1Svg from '../assets/feature1.svg'
import feature2Svg from '../assets/feature2.svg'
import feature3Svg from '../assets/feature3.svg'
import step1Svg from '../assets/step1.svg'
import step2Svg from '../assets/step2.svg'
import step3Svg from '../assets/step3.svg'
import step4Svg from '../assets/step4.svg'
import step5Svg from '../assets/step5.svg'
import step6Svg from '../assets/step6.svg'
import member1Svg from '../assets/1.svg'
import member2Svg from '../assets/2.svg'
import member3Svg from '../assets/3.svg'
import member4Svg from '../assets/4.svg'
import bottomStarSvg from '../assets/landingpage-bottomstar.svg'
import blobsSvg from '../assets/blobs.svg'
import './LandingPage.css'

const LandingPage = () => {
  useEffect(() => {
    if (window.particlesJS) {
      window.particlesJS.load('particles-js', '/particles.json', function () {
        console.log('callback - particles.js config loaded')
      })
    }
  }, [])

  return (
    <div className="landing-page-container">
      <div id="particles-js"></div>
      <Navbar />

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-title-wrapper">
          {/* Oval behind title */}
          <svg
            className="hero-oval"
            viewBox="0 0 520 100"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
          >
            <ellipse
              cx="260"
              cy="50"
              rx="256"
              ry="46"
              stroke="#003660"
              strokeWidth="0.5"
              transform="rotate(3 260 50)"
            />
          </svg>

          <img src={starsSvg} alt="" className="hero-stars" aria-hidden="true" />
          <h1 className="hero-title">
            INTEGRI<span className="hero-title-accent">TREE</span>
          </h1>
        </div>

        <div className="hero-bottom-row">
          <p className="hero-tagline">
            Branching Toward <span className="hero-tagline-accent">Integrity,</span><br />
            Powered by Algorithmic <span className="hero-tagline-accent">Trees.</span>
          </p>

          <Link to="/upload" className="hero-btn-begin">
            <span className="hero-btn-icon">▶</span> Begin
          </Link>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section" id="features-anchor">
        {/* Blue banner header */}
        <div className="features-banner">
          <h2 className="features-banner-title">
            Enabling Transparent E-Wallet Fraud Detection
          </h2>
        </div>

        {/* Cards wrapper */}
        <div className="features-cards-wrapper">
          <div className="feature-card">
            <div className="feature-card-img-wrap">
              <img src={feature1Svg} alt="Fraud Detection" className="feature-card-img" />
            </div>
            <h3 className="feature-card-title">Fraud Detection</h3>
            <p className="feature-card-desc">
              Distinguish fraudulent and legitimate e-wallet transactions using a trained Random Forest model.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-card-img-wrap">
              <img src={feature2Svg} alt="Classification Explanation" className="feature-card-img" />
            </div>
            <h3 className="feature-card-title">Classification Explanation</h3>
            <p className="feature-card-desc">
              Provides SHAP-based explanations showing which transaction feature influenced the most for the model's fraud classification.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-card-img-wrap">
              <img src={feature3Svg} alt="Risk Scoring" className="feature-card-img" />
            </div>
            <h3 className="feature-card-title">Risk Scoring</h3>
            <p className="feature-card-desc">
              Assigns a fraud risk score to each transaction based on the model's predicted fraud probability.
            </p>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section className="how-it-works-section" id="how-it-works-anchor">
        <div className="how-it-works-pill">
          <h2 className="how-it-works-title">Integritree in Action: How it Works</h2>
        </div>

        <div className="how-it-works-grid">
          <div className="how-it-works-item">
            <img src={step1Svg} alt="Step 1" className="step-icon" />
            <h3 className="step-text">
              Upload Transactional<br />Data
            </h3>
          </div>

          <div className="how-it-works-item">
            <img src={step2Svg} alt="Step 2" className="step-icon" />
            <h3 className="step-text">
              Preprocess the<br />Data
            </h3>
          </div>

          <div className="how-it-works-item">
            <img src={step3Svg} alt="Step 3" className="step-icon" />
            <h3 className="step-text">
              Detect Potential<br />Fraud
            </h3>
          </div>

          <div className="how-it-works-item">
            <img src={step4Svg} alt="Step 4" className="step-icon" />
            <h3 className="step-text">
              Review the<br />Results
            </h3>
          </div>

          <div className="how-it-works-item">
            <img src={step5Svg} alt="Step 5" className="step-icon" />
            <h3 className="step-text">
              Explain the<br />Classification
            </h3>
          </div>

          <div className="how-it-works-item">
            <img src={step6Svg} alt="Step 6" className="step-icon" />
            <h3 className="step-text">
              Generate Risk<br />Scores
            </h3>
          </div>
        </div>
      </section>

      {/* Meet the Team Section */}
      <section className="meet-team-section" id="meet-the-team-anchor">
        <h2 className="meet-team-title">Meet the Cultivators Behind Integritree</h2>

        <div className="meet-team-grid">
          {/* Row 1 */}
          <div className="team-card">
            <div className="team-card-info">
              <span className="team-lastname">Cabrera,</span>
              <span className="team-fullname">Chelsea Lauren</span>
            </div>
            <div className="team-avatar-wrap">
              <img src={member1Svg} alt="Chelsea Lauren Cabrera" className="team-avatar" />
            </div>
          </div>

          <div className="team-card">
            <div className="team-avatar-wrap">
              <img src={member2Svg} alt="Joshua Enzo Carpio" className="team-avatar" />
            </div>
            <div className="team-card-info team-card-info--right">
              <span className="team-lastname">Carpio,</span>
              <span className="team-fullname">Joshua Enzo</span>
            </div>
          </div>

          {/* Row 2 */}
          <div className="team-card">
            <div className="team-card-info">
              <span className="team-lastname">Coralde,</span>
              <span className="team-fullname">Jule Alexander</span>
            </div>
            <div className="team-avatar-wrap">
              <img src={member3Svg} alt="Jule Alexander Coralde" className="team-avatar" />
            </div>
          </div>

          <div className="team-card">
            <div className="team-avatar-wrap">
              <img src={member4Svg} alt="Arwen Therese Emmanuelle Matinong" className="team-avatar" />
            </div>
            <div className="team-card-info team-card-info--right">
              <span className="team-lastname">Matinong,</span>
              <span className="team-fullname">Arwen Therese<br />Emmanuelle</span>
            </div>
          </div>
        </div>
      </section>

      {/* Star Divider */}
      <div className="bottom-star-divider">
        <img src={bottomStarSvg} alt="" aria-hidden="true" className="bottom-star-icon" />
      </div>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="landing-footer-content">
          <div className="landing-footer-text">
            <h2 className="landing-footer-heading">Start Analyzing Transactions</h2>
            <p className="landing-footer-sub">
              Begin reviewing e-wallet transactions with fraud detection,<br />
              risk scoring, and classification explanations.
            </p>
          </div>

          <div className="landing-footer-blobs-wrap">
            <img src={blobsSvg} alt="" aria-hidden="true" className="landing-footer-blobs" />
          </div>

          <Link to="/upload" className="landing-footer-btn">
            <span className="hero-btn-icon">▶</span> Begin
          </Link>
        </div>
      </footer>
    </div>
  )
}

export default LandingPage