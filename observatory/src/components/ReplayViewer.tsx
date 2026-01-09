import { FC } from 'react'

import { METTASCOPE_REPLAY_URL_PREFIX } from '../constants'
import { A } from './A'
import { SmallHeader } from './SmallHeader'

export const normalizeReplayUrl = (replayUrl: string | null | undefined): string | null => {
  if (!replayUrl) return null
  if (replayUrl.startsWith(METTASCOPE_REPLAY_URL_PREFIX)) {
    return replayUrl
  }
  return `${METTASCOPE_REPLAY_URL_PREFIX}${replayUrl}`
}

type ReplayViewerProps = {
  replayUrl: string | null | undefined
  label?: string
  height?: number
  showExternalLink?: boolean
}

export const ReplayViewer: FC<ReplayViewerProps> = ({ replayUrl, label, height = 480, showExternalLink = true }) => {
  const normalized = normalizeReplayUrl(replayUrl)
  if (!normalized) {
    return <div className="text-gray-500 text-sm">No replay available.</div>
  }

  return (
    <div className="space-y-2">
      {label ? <SmallHeader>{label}</SmallHeader> : null}
      <div
        className="w-full border border-gray-200 rounded overflow-hidden bg-black"
        style={{ minHeight: '360px', height }}
      >
        <iframe src={normalized} title={label ?? 'Episode replay'} className="w-full h-full" allowFullScreen />
      </div>
      {showExternalLink ? (
        <div className="text-sm">
          <A href={normalized} target="_blank" rel="noopener noreferrer">
            Open replay in new tab
          </A>
        </div>
      ) : null}
    </div>
  )
}
