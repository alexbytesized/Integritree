import React, { useState, useCallback, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { Upload, X, FileText, Database } from 'lucide-react'
import './UploadPage.css'

const UploadPage = () => {
  const [file, setFile] = useState(null)
  const [error, setError] = useState('')
  const [role, setRole] = useState('user') // State to toggle between user and researcher
  const navigate = useNavigate()

  // Initialize particles effect
  useEffect(() => {
    if (window.particlesJS) {
      window.particlesJS.load('particles-js', '/particles.json', function() {
        console.log('callback - particles.js config loaded on upload page');
      });
    }
  }, []);

  const onDrop = useCallback((acceptedFiles, fileRejections) => {
    setError('')
    
    // Check if file was rejected due to type
    if (fileRejections && fileRejections.length > 0) {
      setError('Only CSV format files are allowed.')
      return
    }

    // If the file is an accepted type
    if (acceptedFiles && acceptedFiles.length > 0) {
      const selectedFile = acceptedFiles[0]
      
      // Check the size of the file. Should be <= 500MB
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
      'application/vnd.ms-excel': ['.csv']
    },
    multiple: false
  })

  const removeFile = (e) => {
    setFile(null)
    setError('')
  }

  // Redirects to the results page based on the user (mock purposes)
  const handleAnalyze = () => {
    if (file) {
      if (role === 'researcher') {
        navigate('/results/researcher')
      } else {
        navigate('/results/user')
      }
    }
  }

  // For displaying the uploaded file size on the screen
  const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const dm = decimals < 0 ? 0 : decimals
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
  }

  let cardBodyContent
  if (file) {
    cardBodyContent = (
      <div className="file-details-container">
        <div className="file-info-row">
          <FileText size={24} color="#412D15" />
          <div style={{ textAlign: 'left' }}>
            <div className="file-name">{file.name}</div>
            <div className="file-size">{formatBytes(file.size)}</div>
          </div>
          <button onClick={removeFile} className="btn-remove-file" title="Remove File">
            <X size={16}/>
          </button>
        </div>

        <button onClick={handleAnalyze} className="btn-analyze-submit">
          <span>▶</span> ANALYZE FILE
        </button>
      </div>
    )
  } else {
    cardBodyContent = (
      <div 
        {...getRootProps()} 
        className={`dropzone-area ${isDragActive ? 'drag-active' : ''}`}
      >
        <input {...getInputProps()} />
        <Upload size={32} className="dropzone-icon" />
        <h3 className="dropzone-title">Choose a file or drag and drop it here</h3>
        <p className="dropzone-desc">CSV format only, up to 500MB</p>
        <div className="btn-browse">Browse File</div>
      </div>
    )
  }

  return (
    <div className="upload-page-container">
      <div id="particles-js"></div>

      {/* Title Header */}
      <div className="upload-title-container">
        <h1 className="upload-title">
          INTEGRI<span className="upload-title-accent">TREE</span>
        </h1>
        <p className="upload-subtitle">
          E-Wallet Fraud Detection - Upload File
        </p>
      </div>

      {/* Upload Card */}
      <div className="upload-card-container">
        <div className="upload-card">
          <div className="upload-card-header">
            <div className="upload-header-left">
              <div className="upload-header-icon">
                <Database size={20} />
              </div>
              <div className="upload-header-text">
                <h2 className="upload-card-title">Upload File</h2>
                <p className="upload-card-subtitle">
                  Select and Upload the CSV File containing E-Wallet Transaction Records
                </p>
              </div>
            </div>
            
            {/* User and Researcher Toggle */}
            <div className="role-toggle-container">
              <button 
                type="button"
                className={`role-toggle-btn ${role === 'user' ? 'active' : ''}`}
                onClick={() => setRole('user')}> User
              </button>
              <button 
                type="button"
                className={`role-toggle-btn ${role === 'researcher' ? 'active' : ''}`}
                onClick={() => setRole('researcher')}> Researcher
              </button>
            </div>
          </div>

          <div className="upload-card-divider"></div>

          {/* Upload Card Body */}
          <div className="upload-card-body">
            {cardBodyContent}

            {/* Error Message */}
            {error && <div className="upload-error-msg">{error}</div>}
          </div>
        </div>
      </div>
    </div>
  )
}

export default UploadPage