import clsx from 'clsx'
import { FC } from 'react'
import { Link, LinkProps } from 'react-router-dom'

export const StyledLink: FC<LinkProps & React.RefAttributes<HTMLAnchorElement>> = ({ className, ...props }) => (
  <Link {...props} className={clsx(className, 'text-blue-600 no-underline hover:underline font-medium')} />
)
