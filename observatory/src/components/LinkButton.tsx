import clsx from 'clsx'
import { FC } from 'react'
import { Link } from 'react-router-dom'

import { getButtonClassName } from './Button'

export const LinkButton: FC<{
  to: string
  children: React.ReactNode
  theme?: 'primary' | 'secondary' | 'tertiary'
  type?: 'button' | 'submit'
  size?: 'sm' | 'md'
}> = ({ to, children, theme = 'secondary', type = 'button', size = 'md' }) => {
  return (
    <Link to={to} className={clsx(getButtonClassName(size, theme), 'no-underline')} type={type}>
      {children}
    </Link>
  )
}
