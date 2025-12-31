import { FC, PropsWithChildren } from 'react'

export const Tag: FC<PropsWithChildren> = ({ children }) => {
  return (
    <span className="border border-blue-300 rounded px-2 py-0.5 text-xs bg-blue-50 text-blue-700 leading-none">
      {children}
    </span>
  )
}
