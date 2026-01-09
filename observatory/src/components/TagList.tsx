import { FC } from 'react'

import { Tag } from './Tag'

export const TagList: FC<{ tags: Record<string, string> }> = ({ tags }) => {
  const tagEntries = Object.entries(tags).sort(([a], [b]) => a.localeCompare(b))
  if (tagEntries.length === 0) {
    return null
  }
  return (
    <div className="flex flex-wrap gap-1.5 mt-1.5">
      {tagEntries.map(([key, value]) => (
        <Tag key={`${key}-${value}`}>
          {key}: {value}
        </Tag>
      ))}
    </div>
  )
}
