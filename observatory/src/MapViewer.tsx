interface MapViewerProps {
  selectedEval: string | null
  isViewLocked: boolean
  selectedReplayUrl: string | null
  onToggleLock: () => void
  onReplayClick: () => void
}

// CSS for map viewer
const MAP_VIEWER_CSS = `
.map-viewer {
    position: relative;
    width: 1000px;
    margin: 20px auto;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background: #f9f9f9;
    min-height: 300px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.map-viewer-title {
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eee;
    font-size: 18px;
}
.map-viewer-img {
    max-width: 100%;
    max-height: 350px;
    display: block;
    margin: 0 auto;
}
.map-viewer-placeholder {
    text-align: center;
    color: #666;
    padding: 50px 0;
    font-style: italic;
}
.map-viewer-controls {
    display: flex;
    justify-content: center;
    margin-top: 15px;
    gap: 10px;
}
.map-button {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: #fff;
    cursor: pointer;
    font-size: 14px;
}
.map-button svg {
    width: 14px;
    height: 14px;
}
.map-button.locked {
    background: #f0f0f0;
    border-color: #aaa;
}
.map-button:hover {
    background: #f0f0f0;
}
.map-button.disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
`

const getShortName = (evalName: string) => {
  return evalName.split('/').pop() || evalName
}

const getMapImageUrl = (evalName: string) => {
  if (evalName.toLowerCase() === 'overall') {
    return ''
  }
  const shortName = getShortName(evalName)
  return `https://softmax-public.s3.amazonaws.com/policydash/evals/img/${shortName.toLowerCase()}.png`
}

export function MapViewer({
  selectedEval,
  isViewLocked,
  selectedReplayUrl,
  onToggleLock,
  onReplayClick,
}: MapViewerProps) {
  return (
    <>
      <style>{MAP_VIEWER_CSS}</style>
      <div className="map-viewer">
        <div className="map-viewer-title">{selectedEval || 'Map Viewer'}</div>
        {!selectedEval ? (
          <div className="map-viewer-placeholder">Hover over an evaluation name or cell to see the environment map</div>
        ) : (
          <img
            className="map-viewer-img"
            src={getMapImageUrl(selectedEval)}
            alt={`Environment map for ${selectedEval}`}
            onError={(e) => {
              const target = e.target as HTMLImageElement
              target.style.display = 'none'
              const placeholder = target.parentElement?.querySelector('.map-viewer-placeholder') as HTMLDivElement
              if (placeholder) {
                placeholder.textContent = `No map available for ${selectedEval}`
                placeholder.style.display = 'block'
              }
            }}
          />
        )}

        <div className="map-viewer-controls">
          <button
            type="button"
            className={`map-button ${isViewLocked ? 'locked' : ''}`}
            onClick={onToggleLock}
            title="Lock current view (or click cell)"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <title>Lock icon</title>
              <path
                fillRule="evenodd"
                d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z"
                clipRule="evenodd"
              />
            </svg>
            <span>{isViewLocked ? 'Unlock View' : 'Lock View'}</span>
          </button>
          <button
            type="button"
            className={`map-button ${!selectedReplayUrl ? 'disabled' : ''}`}
            onClick={onReplayClick}
            title="Open replay in Mettascope"
            disabled={!selectedReplayUrl}
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <title>Open replay icon</title>
              <path
                fillRule="evenodd"
                d="M4.25 5.5a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 00.75-.75v-4a.75.75 0 011.5 0v4A2.25 2.25 0 0112.75 17h-8.5A2.25 2.25 0 012 14.75v-8.5A2.25 2.25 0 014.25 4h5a.75.75 0 010 1.5h-5z"
                clipRule="evenodd"
              />
              <path
                fillRule="evenodd"
                d="M6.194 12.753a.75.75 0 001.06.053L16.5 4.44v2.81a.75.75 0 001.5 0v-4.5a.75.75 0 00-.75-.75h-4.5a.75.75 0 000 1.5h2.553l-9.056 8.194a.75.75 0 00-.053 1.06z"
                clipRule="evenodd"
              />
            </svg>
            <span>Open Replay</span>
          </button>
        </div>
      </div>
    </>
  )
}
