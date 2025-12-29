import clsx from 'clsx'
import { FC, PropsWithChildren } from 'react'
import { Link, useLocation } from 'react-router-dom'

const MenuLink: FC<PropsWithChildren<{ to: string; isActive: boolean }>> = ({ to, children, isActive = false }) => {
  return (
    <Link
      to={to}
      className={clsx(
        'py-4 px-5 no-underline border-b-2 transition-all duration-200 hover:bg-gray-100',
        isActive ? 'border-blue-500 text-blue-500' : 'text-gray-500 hover:text-gray-900 border-transparent'
      )}
    >
      {children}
    </Link>
  )
}

export const TopMenu: FC = () => {
  const location = useLocation()

  const isPoliciesActive = location.pathname === '/' || location.pathname.startsWith('/policies')

  return (
    <nav className="border-b border-gray-300 px-5">
      <div className="max-w-7xl mx-auto flex items-center">
        <div className="flex">
          <MenuLink to="/" isActive={isPoliciesActive}>
            Policies
          </MenuLink>
          <MenuLink to="/leaderboard" isActive={location.pathname.startsWith('/leaderboard')}>
            Leaderboard
          </MenuLink>
          <MenuLink to="/eval-tasks" isActive={location.pathname.startsWith('/eval-task')}>
            Remote Jobs
          </MenuLink>
          <MenuLink to="/episode-jobs" isActive={location.pathname.startsWith('/episode-job')}>
            Episode Jobs
          </MenuLink>
          <MenuLink to="/sql-query" isActive={location.pathname === '/sql-query'}>
            SQL Query
          </MenuLink>
        </div>
      </div>
    </nav>
  )
}
