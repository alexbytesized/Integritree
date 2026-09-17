import "./UploadDetails.css"
import backbutton from "../assets/backbutton.png"
import stars from "../assets/stars.png"
import imageplaceholder from "../assets/imageplaceholder.png"
import { useNavigate } from "react-router-dom"

const UploadDetails = () => {
  const navigate = useNavigate()
  return (
    <main className="upload-details-page">
      <header className="upload-details-header">
        <button type="button" className="back-button" aria-label="Go back" onClick={() => navigate(-1)}>
          <img src={backbutton} alt="Back Button" />
        </button>
      </header>

      <section className="screenshot-container">
        <div className="screenshot-title">
          <h1>Upload Details</h1>
          <img src={stars} alt="Stars Icon" />
        </div>

        <div className="transaction-file-name-container">
          <b>CSV File Name:</b>
          <span>dataset.csv</span>
        </div>

        <div className="screenshot-image-container">
          <div className="screenshot-image-inner-container">
            <img src={imageplaceholder} alt="Screenshot" />
          </div>
        </div>
      </section>

      <section className="transaction-details-container">
        <div className="transaction-title">
          <h1>TRANSACTION DETAILS</h1>
        </div>

        <div className="transaction-details-input-container">
          <div className="details-row">
            <label htmlFor="transaction-id">Transaction ID:</label>
            <input id="transaction-id" type="text" defaultValue="12345678" />
          </div>

          <div className="details-row">
            <label htmlFor="amount">Amount:</label>
            <input id="amount" type="text" defaultValue="₱5,000.00" />
          </div>

          <div className="details-row">
            <label htmlFor="transaction-type">Transaction Type:</label>
            <input id="transaction-type" type="text" defaultValue="Transfer" />
          </div>

          <div className="details-row">
            <label htmlFor="transaction-date">Transaction Date:</label>
            <input id="transaction-date" type="text" defaultValue="September 14, 2026" />
          </div>

          <div className="details-row">
            <label htmlFor="transaction-time">Transaction Time:</label>
            <input id="transaction-time" type="text" defaultValue="3:15 AM" />
          </div>

          <div className="details-row">
            <label htmlFor="sender">Sender:</label>
            <input id="sender" type="text" defaultValue="Juan Dela Cruz" />
          </div>

          <div className="details-row">
            <label htmlFor="recipient">Recipient:</label>
            <input id="recipient" type="text" defaultValue="ABC Store" />
          </div>
        </div>

        <div className="proceed-button-container">
          <button type="button" className="proceed-button">
            Proceed
          </button>
        </div>
      </section>
    </main>
  )
}

export default UploadDetails
