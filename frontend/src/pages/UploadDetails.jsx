import { useEffect, useState } from "react"
import { useLocation, useNavigate } from "react-router-dom"

import stars from "../assets/stars.png"
import ReturnButton from "../components/ReturnButton"
import ScreenshotPreview from "../components/ScreenshotPreview"
import TransactionDetails from "../components/TransactionDetails"

import "./UploadDetails.css"

const UploadDetails = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const { fileName = "No file selected", imageUrl = null } = location.state || {}

  const [transactionData, setTransactionData] = useState({
    transactionId: "12345678",
    amount: "5000.00",
    transactionType: "Transfer",
    transactionDate: "September 14, 2026",
    transactionTime: "3:15 AM",
    senderType: "Customer",
    recipient: "Customer",
  })

  useEffect(() => {
    if (window.particlesJS) {
      window.particlesJS.load("particles-js", "/particles.json", function () {})
    }

    return () => {
      if (window.pJSDom && window.pJSDom.length > 0) {
        window.pJSDom.forEach((entry) => {
          if (entry.pJS?.fn?.vendors?.destroypJS) entry.pJS.fn.vendors.destroypJS()
        })
        window.pJSDom = []
      }
    }
  }, [])

  const handleProceed = () => {
    navigate("/user-results", {
      state: {
        fileName,
        imageUrl,
        transactionData,
      },
    })
  }

  return (
    <div className="researcher-results-wrapper">
      <div id="particles-js" className="particles-background" aria-hidden="true" />
      <ReturnButton to="/upload" />

      <main className="user-details-page">
        <section className="user-results-overview">
          <div className="user-overview-title">
            <div className="section-heading">
              <h1>Results Overview</h1>
              <img src={stars} alt="" />
            </div>
          </div>

          <div className="upload-details-content">
            <ScreenshotPreview fileName={fileName} imageUrl={imageUrl} />

            <TransactionDetails initialData={transactionData} onChange={setTransactionData} />

            <div className="proceed-button-container">
              <button type="button" className="proceed-button" onClick={handleProceed}>
                Proceed
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

export default UploadDetails
