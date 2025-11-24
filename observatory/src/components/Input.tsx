import clsx from 'clsx'
import { FC } from 'react'

export const Input: FC<{
  value: string
  onChange: (value: string) => void
  placeholder: string
  size?: 'sm' | 'md'
}> = ({ value, onChange, placeholder, size = 'md' }) => {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={clsx(
        'box-border w-full border border-gray-300',
        size === 'sm' && 'text-xs px-2 py-1 rounded-sm',
        size === 'md' && 'text-sm px-3 py-2 rounded-md'
      )}
    />
  )
}
