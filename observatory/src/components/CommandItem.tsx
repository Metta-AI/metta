import clsx from 'clsx'
import { FC, useState } from 'react'

import { SmallHeader } from './SmallHeader'

type CommandItemProps = {
  label: string
  command: string
  buttonLabel?: string
}

export const CommandItem: FC<CommandItemProps> = ({ label, command, buttonLabel = 'Copy' }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    if (typeof navigator === 'undefined' || typeof navigator.clipboard === 'undefined') {
      return
    }
    try {
      await navigator.clipboard.writeText(command)
      setCopied(true)
      setTimeout(() => setCopied(false), 1000)
    } catch (error) {
      console.error('Failed to copy to clipboard:', error)
    }
  }

  return (
    <div className="flex flex-col">
      <SmallHeader>{label}</SmallHeader>
      <button
        type="button"
        className={clsx(
          'flex items-center gap-3 px-3 py-2 rounded-md border text-left font-mono',
          'transition-colors cursor-pointer',
          copied ? 'border-blue-700 bg-indigo-50' : 'border-blue-200 bg-slate-50 hover:border-slate-400'
        )}
        onClick={handleCopy}
      >
        <code className="flex-1 text-xs text-slate-900 wrap-break-word">{command}</code>
        <span className="text-xs font-semibold text-blue-700 whitespace-nowrap">
          {copied ? 'Copied!' : buttonLabel}
        </span>
      </button>
    </div>
  )
}
