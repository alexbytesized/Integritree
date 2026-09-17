import React, { useState, useCallback, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { ArrowLeft, Database, Upload, X, FileText } from 'lucide-react'
import './UploadPage.css'

const ResearcherUploadPage = () => {
  const [file, setFile] = useState(null)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    if (window.particlesJS) {
      window.particlesJS.load('particles-js', '/particles.json', () => {})
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
        setError('File size exceeds the 500MB limit.')
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

  const handleAnalyze = () => {
    if (file) {
      navigate('/researcher')
    }
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
    <div className="upload-page-container">
      <div id="particles-js"></div>

      {/* Back Button */}
      <Link to="/" className="upload-back-btn">
        <ArrowLeft size={20} strokeWidth={2} />
      </Link>

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

        {/* Card Header */}
        <div className="upload-card-header">
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
                <button onClick={removeFile} className="btn-remove-file" title="Remove File">
                  <X size={16} />
                </button>
              </div>

              <button onClick={handleAnalyze} className="btn-analyze-submit">
                <span>▶</span> ANALYZE FILE
              </button>
            </div>
          ) : (
            <div
              {...getRootProps()}
              className={`dropzone-area ${isDragActive ? 'drag-active' : ''}`}
            >
              <input {...getInputProps()} />
              <Upload size={36} className="dropzone-icon" />
              <h3 className="dropzone-title">Choose a file or drag and drop it here</h3>
              <p className="dropzone-desc">CSV format only, up to 500MB</p>
              <div className="btn-browse">Browse File</div>
            </div>
          )}

          {error && <div className="upload-error-msg">{error}</div>}
        </div>
      </div>
    </div>
  )
}

export default ResearcherUploadPage
