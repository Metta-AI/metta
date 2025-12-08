export const parseDatetime = (value: string | null): Date | null => {
  if (!value) {
    return null
  }
  const hasTimezone = value.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(value)
  const normalizedValue = hasTimezone ? value : `${value}Z`
  const date = new Date(normalizedValue)
  if (Number.isNaN(date.getTime())) {
    return null
  }
  return date
}

export const formatDate = (value: string | null): string => {
  const date = parseDatetime(value)
  if (!date) {
    return '—'
  }
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

export const formatRelativeTime = (value: string | null): string => {
  const date = parseDatetime(value)
  if (!date) {
    return '—'
  }

  const diffMs = Date.now() - date.getTime()
  const diffSeconds = Math.max(0, Math.floor(diffMs / 1000))

  if (diffSeconds < 120) return 'just now'
  const diffMinutes = Math.floor(diffSeconds / 60)
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  const diffHours = Math.floor(diffMinutes / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  const diffDays = Math.floor(diffHours / 24)
  if (diffDays < 7) return `${diffDays}d ago`
  const diffWeeks = Math.floor(diffDays / 7)
  if (diffWeeks < 4) return `${diffWeeks}w ago`
  const diffMonths = Math.floor(diffDays / 30)
  if (diffMonths < 12) return `${diffMonths}mo ago`
  const diffYears = Math.floor(diffDays / 365)
  return `${diffYears}y ago`
}

export const formatDurationBetween = (from: string | null, to: string | null): string | null => {
  const fromDate = parseDatetime(from)
  const toDate = parseDatetime(to)
  if (!fromDate || !toDate) return null
  const diffMs = toDate.getTime() - fromDate.getTime()
  if (diffMs < 0) return null
  const seconds = Math.floor(diffMs / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  if (minutes < 60) return `${minutes}m ${remainingSeconds}s`
  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60
  return `${hours}h ${remainingMinutes}m`
}
