import clsx from 'clsx'

export const A: React.FC<React.AnchorHTMLAttributes<HTMLAnchorElement>> = ({ className, children, ...props }) => {
  return (
    <a {...props} className={clsx(className, 'text-blue-600 no-underline hover:underline')}>
      {children}
    </a>
  )
}
