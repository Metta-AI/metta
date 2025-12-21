import clsx from 'clsx'
import { FC } from 'react'

import { EvalStatusInfo } from './LeaderboardEntry'

export const PolicyStatusBadge: FC<{ status: EvalStatusInfo }> = ({ status }) => {
  return (
    <span
      className={clsx(
        'rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase border tracking-wide',
        status.status === 'pending' && 'border-slate-300 text-slate-600 bg-slate-100',
        status.status === 'complete' && 'border-green-300 text-green-700 bg-green-100',
        status.status === 'canceled' && 'border-red-200 text-red-700 bg-red-100'
      )}
    >
      {status.status}
    </span>
  )
}
