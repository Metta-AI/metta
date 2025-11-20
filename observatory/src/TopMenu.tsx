import clsx from 'clsx'
import { FC, PropsWithChildren } from 'react'
import { Link, useLocation } from 'react-router-dom'

// CSS for navigation
const NAV_CSS = `
.nav-tab {
  padding: 15px 20px;
  text-decoration: none;
  color: #666;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}

.nav-tab:hover {
  color: #333;
  background: #f8f9fa;
}

.nav-tab.active {
  color: #007bff;
  border-bottom-color: #007bff;
}
`

const MenuLink: FC<PropsWithChildren<{ to: string; isActive: boolean }>> = ({ to, children, isActive = false }) => {
  return (
    <Link to={to} className={clsx('nav-tab', isActive && 'active')}>
      {children}
    </Link>
  )
}

export const TopMenu: FC = () => {
  const location = useLocation()

  return (
    <>
      <style>{NAV_CSS}</style>
      <nav className="border-b border-gray-300 px-5">
        <div className="max-w-7xl mx-auto flex items-center">
          <div className="flex">
            <MenuLink to="/eval-tasks" isActive={location.pathname.startsWith('/eval-task')}>
              Evaluate Policies
            </MenuLink>
            <MenuLink to="/leaderboard" isActive={location.pathname === '/leaderboard'}>
              Leaderboard
            </MenuLink>
            <MenuLink to="/sql-query" isActive={location.pathname === '/sql-query'}>
              SQL Query
            </MenuLink>
          </div>
        </div>
      </nav>
    </>
  )
}
