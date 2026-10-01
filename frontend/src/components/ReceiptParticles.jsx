import { useEffect, useRef, useId } from 'react'

export default function ReceiptParticles() {
  const container = useRef(null)
  const reactId = useId()
  const canvasId = `particles-${reactId.replace(/:/g, '')}`

  useEffect(() => {
    const abort = new AbortController()
    let instance, observer, frame
    const resize = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        if (!instance || !instance.pJS) return
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
      window.particlesJS(canvasId, config)
      instance = (window.pJSDom || []).find(entry => entry?.pJS?.canvas?.el?.parentElement === container.current)
      observer = new ResizeObserver(resize)
      observer.observe(container.current)
    }).catch(() => {})
    return () => {
      abort.abort()
      observer?.disconnect()
      cancelAnimationFrame(frame)
      if (instance?.pJS?.canvas?.el) {
        cancelAnimationFrame(instance.pJS.fn.drawAnimFrame)
        instance.pJS.canvas.el.remove()
        window.pJSDom = (window.pJSDom || []).filter(entry => entry !== instance)
      }
    }
  }, [canvasId])
  return <div id={canvasId} ref={container} className="particles-background" aria-hidden="true" />
}

