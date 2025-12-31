import clsx from 'clsx'
import { createContext, FC, PropsWithChildren, use } from 'react'

type TableTheme = 'large' | 'inner' | 'light'

const TableContext = createContext<{ theme: TableTheme }>({ theme: 'large' })

export const Table: FC<PropsWithChildren<{ theme?: TableTheme }>> & {
  Header: FC<PropsWithChildren>
  Body: FC<PropsWithChildren>
} = ({ children, theme = 'large' }) => {
  return (
    <TableContext value={{ theme }}>
      <table
        className={clsx(
          'w-full border-collapse',
          theme === 'large' && 'text-sm',
          theme === 'inner' && 'text-xs',
          theme === 'light' && 'text-sm'
        )}
      >
        {children}
      </table>
    </TableContext>
  )
}

export const TableHeader: FC<PropsWithChildren> = ({ children }) => {
  return <thead className="bg-gray-50">{children}</thead>
}

export const TableBody: FC<PropsWithChildren> = ({ children }) => {
  return <tbody>{children}</tbody>
}

export const TH: FC<React.ThHTMLAttributes<HTMLTableCellElement>> = ({ className, ...props }) => {
  const { theme } = use(TableContext)

  return (
    <th
      className={clsx(
        'text-left text-xs font-bold tracking-[0.04em] uppercase text-gray-600 align-top',
        theme === 'large' && 'px-3 py-2 border-b border-gray-200',
        theme === 'inner' && 'p-1.5 border border-gray-200 text-gray-500 bg-gray-100',
        theme === 'light' && 'p-2',
        className
      )}
      {...props}
    />
  )
}

export const TR: FC<React.HTMLAttributes<HTMLTableRowElement>> = ({ ...props }) => {
  return <tr {...props} />
}

export const TD: FC<PropsWithChildren<React.TdHTMLAttributes<HTMLTableCellElement>>> = ({
  children,
  className,
  ...props
}) => {
  const { theme } = use(TableContext)
  return (
    <td
      className={clsx(
        // TODO - use tailwind-merge or tailwind-variants?
        'align-top',
        theme === 'large' && 'px-3 py-2 border-b border-gray-100',
        theme === 'inner' && 'p-1.5 border border-gray-200',
        theme === 'light' && 'p-2',
        className
      )}
      {...props}
    >
      {children}
    </td>
  )
}

Table.Header = TableHeader
Table.Body = TableBody
