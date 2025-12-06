import { clsx } from 'clsx'
import { type FC } from 'react'

export type PaginationState = {
  page: number
  limit: number
  total: number
}

type Props = {
  className?: string
  loading?: boolean
  state: PaginationState
  onNextFn: () => void
  onPrevFn: () => void
}

const PaginationButton: FC<{
  disabled: boolean
  label: string
  loading?: boolean
  onClickFn: () => void
}> = ({ disabled, label, loading = false, onClickFn }) => {
  const color = disabled ? 'bg-gray-300 text-gray-500' : 'bg-blue-500 hover:bg-blue-600 text-white'
  const cursor = disabled ? 'cursor-not-allowed' : 'cursor-pointer'
  return (
    <button
      className={clsx('px-3 py-1 border-0 ring-0 rounded', color, cursor)}
      disabled={disabled}
      onClick={() => (loading ? null : onClickFn())}
    >
      {label}
    </button>
  )
}

export const Pagination: FC<Props> = ({ className = '', loading = false, state, onNextFn, onPrevFn }) => {
  const totalPages = Math.ceil(state.total / state.limit)
  return (
    <div className={clsx(className, 'flex items-center gap-1')}>
      <div className="mx-4 text-xs text-gray-700">
        Page {state.page} of {totalPages} ({state.total} items)
      </div>
      <PaginationButton disabled={state.page <= 1} label="Previous" loading={loading} onClickFn={onPrevFn} />
      <PaginationButton disabled={state.page >= totalPages} label="Next" loading={loading} onClickFn={onNextFn} />
    </div>
  )
}
