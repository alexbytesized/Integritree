import { Routes, Route } from "react-router-dom"
import ReceiptLifecycle from "./components/ReceiptLifecycle"
import LandingPage from "./pages/LandingPage"
import UploadPage from "./pages/UploadPage"
import ResearcherUploadPage from "./pages/ResearcherUploadPage"
import ResearcherResultsPage from "./pages/ResearcherResultsPage"
import TransactionDetailsPage from "./pages/TransactionDetailsPage"
import UploadDetails from "./pages/UploadDetails"
import UserResultsPage from "./pages/UserResultsPage"

const App = () => {
  return (
    <ReceiptLifecycle>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/researcher-upload" element={<ResearcherUploadPage />} />
        <Route path="/researcher" element={<ResearcherResultsPage />} />
        <Route path="/researcher/transaction/:id" element={<TransactionDetailsPage />} />
        <Route path="/results" element={<UploadDetails />} />
        <Route path="/user-results" element={<UserResultsPage />} />
      </Routes>
    </ReceiptLifecycle>
  )
}

export default App
