import { useEffect } from "react"
import { useNavigate } from "react-router-dom"

import backbutton from "../assets/back-button_revised.png"
import stars from "../assets/stars.png"

import "./UploadDetails.css"

import ScreenshotPreview from "../components/ScreenshotPreview"

import TransactionDetails from "../components/TransactionDetails"

const UploadDetails = () => {
  const navigate = useNavigate()

  useEffect(() => {
    if (window.particlesJS) {
      window.particlesJS.load("particles-js", "/particles.json", function () {
        console.log("callback - particles.js config loaded")
      })
    }
  }, [])

  const handleProceed = () => {
    console.log("Proceed clicked")
  }

  return (
    <>
      <div id="particles-js" className="particles-background" aria-hidden="true" />

      <main className="user-details-page">
        <header className="user-results-header">
          <button type="button" className="back-button" aria-label="Go back" onClick={() => navigate(-1)}>
            <img src={backbutton} alt="Back" />
          </button>
        </header>

        <section className="user-results-overview">
          <div className="user-overview-title">
            <div className="section-heading">
              <h1>Results Overview</h1>
              <img src={stars} alt="" />
            </div>
          </div>

          <div className="upload-details-content">
            <ScreenshotPreview fileName="Test" />

            <TransactionDetails />

            <div className="proceed-button-container">
              <button type="button" className="proceed-button" onClick={handleProceed}>
                Proceed
              </button>
            </div>
          </div>
        </section>
      </main>
    </>
  )
}

export default UploadDetails
