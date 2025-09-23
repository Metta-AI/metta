import { useState } from 'react'
import { SearchInput } from './components/SearchInput'
import { CreateTaskForm } from './components/CreateTaskForm'
import { TaskTable } from './components/TaskTable'
import { useEvalTasks } from './hooks/useEvalTasks'
import { EvalTasksProps } from './types/evalTasks'

const TabButton = ({
  isActive,
  onClick,
  children,
  count,
}: {
  isActive: boolean
  onClick: () => void
  children: React.ReactNode
  count: number
}) => (
  <button
    onClick={onClick}
    style={{
      padding: '12px 24px',
      backgroundColor: isActive ? '#007bff' : '#f8f9fa',
      color: isActive ? 'white' : '#495057',
      borderRadius: '8px 8px 0 0',
      cursor: 'pointer',
      fontSize: '16px',
      fontWeight: isActive ? '600' : '400',
      transition: 'all 0.2s ease',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      border: `1px solid ${isActive ? '#007bff' : '#dee2e6'}`,
      borderBottom: isActive ? '1px solid #007bff' : '1px solid #dee2e6',
      marginRight: '4px',
    }}
    onMouseEnter={(e) => {
      if (!isActive) {
        e.currentTarget.style.backgroundColor = '#e9ecef'
      }
    }}
    onMouseLeave={(e) => {
      if (!isActive) {
        e.currentTarget.style.backgroundColor = '#f8f9fa'
      }
    }}
  >
    {children}
    <span
      style={{
        backgroundColor: isActive ? 'rgba(255,255,255,0.2)' : '#6c757d',
        color: isActive ? 'white' : 'white',
        borderRadius: '12px',
        padding: '2px 8px',
        fontSize: '12px',
        fontWeight: '600',
      }}
    >
      {count}
    </span>
  </button>
)

export function EvalTasks({ repo }: EvalTasksProps) {
  const {
    tasks,
    activeSortField,
    activeSortDirection,
    completedSortField,
    completedSortDirection,
    expandedRows,
    loadTasks,
    handleSort,
    toggleRowExpansion,
    activeTasks,
    historyTasks,
    sortedActiveTasks,
    sortedHistoryTasks,
  } = useEvalTasks(repo)

  const [searchText, setSearchText] = useState('')
  const [searchLoading, setSearchLoading] = useState(false)
  const [currentView, setCurrentView] = useState<'active' | 'history'>('active')

  const handleSearchChange = (value: string) => {
    setSearchText(value)
    const searchValue = value.trim() || undefined
    setSearchLoading(true)
    loadTasks(searchValue)
    setSearchLoading(false)
  }

  const currentTasks = currentView === 'active' ? sortedActiveTasks : sortedHistoryTasks
  const currentCount = currentView === 'active' ? activeTasks.length : historyTasks.length

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '30px' }}>Policy Evaluation Tasks</h1>

      {/* Create Task Form */}
      <CreateTaskForm repo={repo} tasks={tasks} onTaskCreated={async () => loadTasks()} />

      {/* Search Section */}
      <div
        style={{
          backgroundColor: '#ffffff',
          padding: '20px',
          borderRadius: '12px',
          marginBottom: '20px',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
          border: '1px solid #e8e8e8',
        }}
      >
        <h3 style={{ marginBottom: '30px' }}>Search</h3>
        <SearchInput searchText={searchText} onSearchChange={handleSearchChange} />
        {searchLoading && <div style={{ marginTop: '8px', fontSize: '14px', color: '#6c757d' }}>Searching...</div>}
      </div>

      {/* Tab Navigation */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <TabButton
            isActive={currentView === 'active'}
            onClick={() => setCurrentView('active')}
            count={activeTasks.length}
          >
            Active Tasks
          </TabButton>
          <TabButton
            isActive={currentView === 'history'}
            onClick={() => setCurrentView('history')}
            count={historyTasks.length}
          >
            History
          </TabButton>
        </div>
      </div>

      {/* Current View Section */}
      <div
        style={{
          backgroundColor: '#ffffff',
          border: '1px solid #dee2e6',
          borderRadius: '0 8px 8px 8px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            padding: '16px 20px',
            backgroundColor: '#f8f9fa',
            borderBottom: '1px solid #dee2e6',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <h2 style={{ margin: '0', fontSize: '18px' }}>
            {currentView === 'active' ? 'Active Tasks' : 'Task History'} ({currentCount})
            {searchText && (
              <span style={{ fontSize: '14px', fontWeight: 'normal', color: '#6c757d', marginLeft: '8px' }}>
                (filtered by "{searchText}")
              </span>
            )}
          </h2>
        </div>

        <div style={{ padding: '0' }}>
          <TaskTable
            tasks={currentTasks}
            isActive={currentView === 'active'}
            expandedRows={expandedRows}
            onToggleExpansion={toggleRowExpansion}
            onSort={handleSort}
            activeSortField={activeSortField}
            activeSortDirection={activeSortDirection}
            completedSortField={completedSortField}
            completedSortDirection={completedSortDirection}
          />
        </div>
      </div>
    </div>
  )
}
