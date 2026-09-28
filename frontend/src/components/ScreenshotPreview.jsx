import "./ScreenshotPreview.css"
import cameraIcon from "../assets/imageplaceholder.png"

const ScreenshotPreview = ({ fileName, imageUrl }) => {
  return (
    <div className="screenshot-preview">
      <div className="screenshot-preview-header">
        <span>Screenshot File Name:</span>
        <span>{fileName}</span>
      </div>

      <div className="screenshot-preview-content">
        <div className="screenshot-preview-image-container">
          {imageUrl ? (
            <img className="screenshot-preview-image" src={imageUrl} alt="Uploaded transaction screenshot" />
          ) : (
            <div className="screenshot-placeholder">
              <img src={cameraIcon} alt="" />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ScreenshotPreview
