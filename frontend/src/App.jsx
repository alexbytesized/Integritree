import React from "react"
import { Routes, Route } from "react-router-dom"
import LandingPage from "./pages/LandingPage"
import UploadPage from "./pages/UploadPage"
import ResearcherResultsPage from "./pages/ResearcherResultsPage"
import UploadDetails from "./pages/UploadDetails"

const App = () => {
  return (
    <div>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/researcher" element={<ResearcherResultsPage />} />
      </Routes>
    </div>
  )
}

export default App
