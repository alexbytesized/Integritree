import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import particlesUrl from 'particles.js/particles.js?url'

// particles.js 2 uses legacy non-strict JavaScript; load its local asset as a classic script.
const particlesScript = document.createElement('script')
particlesScript.src = particlesUrl
const ready = new Promise(resolve => { particlesScript.onload = resolve; particlesScript.onerror = resolve })
document.head.appendChild(particlesScript)
ready.then(() => createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <App />
  </BrowserRouter>,
))