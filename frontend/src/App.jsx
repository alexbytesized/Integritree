import React from 'react'
import { Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import UploadPage from './pages/UploadPage'
import UserResultsPage from './pages/UserResultsPage'
import ResearcherResultsPage from './pages/ResearcherResultsPage'

const App = () => {
  return (
    <div>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/results" element={<UserResultsPage />} />
        <Route path="/researcher" element={<ResearcherResultsPage />} />
      </Routes>
    </div>
  )
}

export default App