import { FC, PropsWithChildren } from 'react'

export const Card: FC<
  PropsWithChildren<{
    title?: string
  }>
> = ({ children, title }) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
      {title && (
        <div className="px-5 py-2 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 my-2">{title}</h2>
          </div>
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  )
}
