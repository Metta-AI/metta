import { useState } from 'react'
import { SearchInput } from './components/SearchInput'
import { CreateTaskForm } from './components/CreateTaskForm'
import { TaskTable } from './components/TaskTable'
import { useEvalTasks } from './hooks/useEvalTasks'
import { EvalTasksProps } from './types/evalTasks'

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

  const handleSearchChange = (value: string) => {
    setSearchText(value)

    // Don't block the input - search in background
    const searchValue = value.trim() || undefined
    setSearchLoading(true)

    loadTasks(searchValue)
      .catch((error) => console.error('Search failed:', error))
      .finally(() => setSearchLoading(false))
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '30px' }}>Policy Evaluation Tasks</h1>

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
        <SearchInput searchText={searchText} onSearchChange={handleSearchChange} />
        {searchLoading && <div style={{ marginTop: '8px', fontSize: '14px', color: '#6c757d' }}>Searching...</div>}
      </div>

      {/* Create Task Form */}
      <CreateTaskForm repo={repo} tasks={tasks} onTaskCreated={loadTasks} />

      {/* Active Section */}
      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ marginBottom: '20px' }}>
          Active ({activeTasks.length})
          {searchText && (
            <span style={{ fontSize: '14px', fontWeight: 'normal', color: '#6c757d', marginLeft: '8px' }}>
              (filtered by "{searchText}")
            </span>
          )}
        </h2>
        <TaskTable
          tasks={sortedActiveTasks}
          isActive={true}
          expandedRows={expandedRows}
          onToggleExpansion={toggleRowExpansion}
          onSort={handleSort}
          activeSortField={activeSortField}
          activeSortDirection={activeSortDirection}
          completedSortField={completedSortField}
          completedSortDirection={completedSortDirection}
        />
      </div>

      {/* History Section */}
      <div>
        <h2 style={{ marginBottom: '20px' }}>
          History ({historyTasks.length})
          {searchText && (
            <span style={{ fontSize: '14px', fontWeight: 'normal', color: '#6c757d', marginLeft: '8px' }}>
              (filtered by "{searchText}")
            </span>
          )}
        </h2>
        <TaskTable
          tasks={sortedHistoryTasks}
          isActive={false}
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
  )
}
