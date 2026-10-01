import { useState, useCallback, useEffect } from 'react'
import { uploadCsv } from '../researchApi'
import { useNavigate, Link } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { Database, Upload, X, FileText, Info } from 'lucide-react'
import ReturnButton from '../components/ReturnButton'
import ResearcherInstructionsModal from '../components/ResearcherInstructionsModal'
import './UploadPage.css'

const ResearcherUploadPage = () => {
  const [file, setFile] = useState(null)
  const [error, setError] = useState('')
  const [progress, setProgress] = useState(null)
  const [busy, setBusy] = useState(false)
  const [showInstructions, setShowInstructions] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    if (window.particlesJS) {
      window.particlesJS.load('particles-js', '/particles.json', () => {})
    }

    return () => {
      if (window.pJSDom && window.pJSDom.length > 0) {
        window.pJSDom.forEach((entry) => {
          if (entry.pJS && entry.pJS.fn && entry.pJS.fn.vendors && entry.pJS.fn.vendors.destroypJS) {
            entry.pJS.fn.vendors.destroypJS()
          }
        })
        window.pJSDom = []
      }
    }
  }, [])

  const onDrop = useCallback((acceptedFiles, fileRejections) => {
    setError('')

    if (fileRejections && fileRejections.length > 0) {
      setError('Only CSV format files are allowed.')
      return
    }

    if (acceptedFiles && acceptedFiles.length > 0) {
      const selectedFile = acceptedFiles[0]

      if (selectedFile.size > 500 * 1024 * 1024) {
        setError('File size exceeds the 500 MiB limit.')
        return
      }

      setFile(selectedFile)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv'],
    },
    multiple: false,
  })

  const removeFile = () => {
    setFile(null)
    setError('')
  }

  const handleAnalyze = async () => {
    if (!file || busy) return
    setBusy(true)
    setError('')
    try {
      const result = await uploadCsv(file, setProgress)
      navigate(`/researcher?analysis=${result.id}`)
    } catch (err) { setError(err.message) }
    finally { setBusy(false) }
  }

  const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const dm = decimals < 0 ? 0 : decimals
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
  }

  return (
    <div className="upload-page-container researcher-upload">
      <div id="particles-js"></div>

      {/* Back Button */}
      <ReturnButton />

      {/* Title Header */}
      <div className="upload-title-container">
        <h1 className="upload-logo">
          INTEGRI<span className="upload-logo-accent">TREE</span>
        </h1>
        <p className="upload-subtitle">E-Wallet Fraud Detection - Upload File</p>
        <p className="upload-researcher-link">
          For Users?{' '}
          <Link to="/upload" className="upload-researcher-anchor">
            Click here
          </Link>
        </p>
      </div>

      {/* Upload Card */}
      <div className="upload-card">
        {new URLSearchParams(window.location.search).has('expired') && <p role="alert">Your backend session expired. Upload the CSV again to start a new analysis.</p>}

        {/* Card Header */}
        <div className="upload-card-header">
          <div className="upload-header-left">
            <div className="upload-header-icon">
              <Database size={22} strokeWidth={1.5} />
            </div>
            <div className="upload-header-text">
              <h2 className="upload-card-title">Upload File</h2>
              <p className="upload-card-subtitle">
                Select and Upload the CSV File containing E-Wallet Transaction Records
              </p>
            </div>
          </div>
          <button
            type="button"
            className="btn-upload-instructions"
            title="Upload Instructions"
            onClick={() => setShowInstructions(true)}
          >
            <Info size={16} />
            Upload Instructions
          </button>
        </div>

        {/* Dropzone / File Area */}
        <div className="upload-card-body">
          {file ? (
            <div className="file-details-container">
              <div className="file-info-row">
                <FileText size={24} color="#1a73e8" />
                <div style={{ textAlign: 'left', flex: 1 }}>
                  <div className="file-name">{file.name}</div>
                  <div className="file-size">{formatBytes(file.size)}</div>
                </div>
                <button disabled={busy} onClick={removeFile} className="btn-remove-file" title="Remove File">
                  <X size={16} />
                </button>
              </div>

              <button disabled={busy} onClick={handleAnalyze} className="btn-analyze-submit">
                <span>▶</span> {busy ? 'ANALYZING FILE…' : 'ANALYZE FILE'}
              </button>

              {busy && (
                <p className="upload-status-text" role="status" aria-live="polite">
                  Uploading: {progress ?? 0}%
                </p>
              )}
            </div>
          ) : (
            <div
              {...getRootProps()}
              className={`dropzone-area ${isDragActive ? 'drag-active' : ''}`}
            >
              <input {...getInputProps()} />
              <Upload size={36} className="dropzone-icon" />
              <h3 className="dropzone-title">Choose a file or drag and drop it here</h3>
              <p className="dropzone-desc">CSV format only, up to 500 MB. No fixed record-count limit.</p>
              <div className="btn-browse">Browse File</div>
            </div>
          )}

          {error && <div className="upload-error-msg" role="alert">{error}</div>}
        </div>
      </div>

      {showInstructions && (
        <ResearcherInstructionsModal onClose={() => setShowInstructions(false)} />
      )}
    </div>
  )
}

export default ResearcherUploadPage
