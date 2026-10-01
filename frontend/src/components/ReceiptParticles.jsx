import { useEffect, useRef } from 'react'

export default function ReceiptParticles() {
  const container = useRef(null)
  useEffect(() => {
    const abort = new AbortController()
    let instance, observer, frame
    const resize = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        if (!instance) return
        const { canvas, tmp, fn } = instance.pJS
        const scale = tmp.retina ? canvas.pxratio : 1
        const width = canvas.el.offsetWidth * scale
        const height = canvas.el.offsetHeight * scale
        if (canvas.w === width && canvas.h === height) return
        canvas.w = canvas.el.width = width
        canvas.h = canvas.el.height = height
        fn.vendors.densityAutoParticles()
      })
    }
    // Use the unchanged home-page configuration; resize when async content changes height.
    fetch('/particles.json', { signal: abort.signal }).then(response => response.json()).then(config => {
      if (abort.signal.aborted || !container.current || !window.particlesJS) return
      window.particlesJS('receipt-particles', config)
      instance = window.pJSDom.find(entry => entry.pJS.canvas.el.parentElement === container.current)
      observer = new ResizeObserver(resize)
      observer.observe(container.current)
    }).catch(() => {})
    return () => {
      abort.abort()
      observer?.disconnect()
      cancelAnimationFrame(frame)
      if (instance) {
        cancelAnimationFrame(instance.pJS.fn.drawAnimFrame)
        instance.pJS.canvas.el.remove()
        window.pJSDom = (window.pJSDom || []).filter(entry => entry !== instance)
      }
    }
  }, [])
  return <div id="receipt-particles" ref={container} className="particles-background" aria-hidden="true" />
}
