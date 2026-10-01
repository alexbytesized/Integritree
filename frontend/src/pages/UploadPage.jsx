import { useState, useCallback, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { Camera, ImageIcon, X, FileImage } from 'lucide-react'
import ReturnButton from '../components/ReturnButton'
import './UploadPage.css'
import { uploadReceipt, receiptKey } from '../receiptApi'

const UploadPage = () => {
  const [file, setFile] = useState(null)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
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
      setError('Only PNG/JPG format files are allowed.')
      return
    }

    if (acceptedFiles && acceptedFiles.length > 0) {
      const selectedFile = acceptedFiles[0]

      if (selectedFile.size > 10 * 1024 * 1024) {
        setError('File size exceeds the 10 MiB limit.')
        return
      }

      setFile(selectedFile)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
    },
    multiple: false,
    disabled: uploading,
  })

  const removeFile = () => {
    setFile(null)
    setError('')
  }

  const handleAnalyze = async () => {
    if (!file || uploading) return
    setUploading(true)
    setError('')
    try {
      const job = await uploadReceipt(file)
      navigate(`/results?receipt=${job.id}`)
    } catch (err) { setError(err.message) }
    finally { setUploading(false) }
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
      <ReturnButton />

      {/* Title Header */}
      <div className="upload-title-container">
        <h1 className="upload-logo">
          INTEGRI<span className="upload-logo-accent">TREE</span>
        </h1>
        <p className="upload-subtitle">E-Wallet Fraud Detection - Upload File</p>
        <p className="upload-researcher-link">
          For Researchers?{' '}
          <Link to="/researcher-upload" className="upload-researcher-anchor">
            Click here
          </Link>
        </p>
        <p>Upload GCash Express Send, Pay Online, or bank-account transfer screenshots. You can edit all five transaction types after extraction; Cash In/Out and QR screenshot layouts are not supported yet.</p>
        {sessionStorage.getItem(receiptKey) && <p><Link to={`/results?receipt=${sessionStorage.getItem(receiptKey)}`}>Resume or clear your current receipt</Link></p>}
      </div>

      {/* Upload Card */}
      <div className="upload-card">

        {/* Card Header */}
        <div className="upload-card-header">
          <div className="upload-header-icon">
            <Camera size={22} strokeWidth={1.5} />
          </div>
          <div className="upload-header-text">
            <h2 className="upload-card-title">Upload Screenshot</h2>
            <p className="upload-card-subtitle">
              Select and Upload the screenshot of your transaction record
            </p>
          </div>
        </div>

        {/* Dropzone / File Area */}
        <div className="upload-card-body">
          {file ? (
            <div className="file-details-container">
              <div className="file-info-row">
                <FileImage size={24} color="#1a73e8" />
                <div style={{ textAlign: 'left', flex: 1 }}>
                  <div className="file-name">{file.name}</div>
                  <div className="file-size">{formatBytes(file.size)}</div>
                </div>
                <button onClick={removeFile} disabled={uploading} className="btn-remove-file" title="Remove File">
                  <X size={16} />
                </button>
              </div>

              <button onClick={handleAnalyze} disabled={uploading} className="btn-analyze-submit">
                {uploading ? 'Uploading…' : 'Extract transaction details'}
              </button>
            </div>
          ) : (
            <div
              {...getRootProps()}
              className={`dropzone-area ${isDragActive ? 'drag-active' : ''}`}
            >
              <input {...getInputProps()} />
              <ImageIcon size={36} className="dropzone-icon" />
              <h3 className="dropzone-title">Choose a file or drag and drop it here</h3>
              <p className="dropzone-desc">One PNG/JPG, up to 10 MiB and 20 million pixels</p>
              <div className="btn-browse">Browse File</div>
            </div>
          )}

          {error && <div className="upload-error-msg" role="alert">{error}</div>}
          <p>Review and confirm extracted details before prediction. Receipts clear automatically when you leave the receipt flow or close this tab. Refresh preserves your current receipt.</p>
        </div>
      </div>
    </div>
  )
}

export default UploadPage
