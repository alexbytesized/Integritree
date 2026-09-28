import React from "react"
import { Routes, Route } from "react-router-dom"
import LandingPage from "./pages/LandingPage"
import UploadPage from "./pages/UploadPage"
import ResearcherUploadPage from "./pages/ResearcherUploadPage"
import ResearcherResultsPage from "./pages/ResearcherResultsPage"
import TransactionDetailsPage from "./pages/TransactionDetailsPage"
import UploadDetails from "./pages/UploadDetails"

const App = () => {
  return (
    <div>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/researcher-upload" element={<ResearcherUploadPage />} />
        <Route path="/researcher" element={<ResearcherResultsPage />} />
        <Route path="/researcher/transaction/:id" element={<TransactionDetailsPage />} />
        <Route path="/results" element={<UploadDetails />} />
      </Routes>
    </div>
  )
}

export default App
