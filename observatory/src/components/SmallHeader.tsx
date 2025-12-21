import { FC, PropsWithChildren } from 'react'

export const SmallHeader: FC<PropsWithChildren> = ({ children }) => {
  return <div className="text-xs font-semibold text-slate-600 uppercase tracking-wide mb-1">{children}</div>
}
