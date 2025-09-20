import { useState, useEffect, useCallback } from 'react'
import { EvalTask } from '../repo'
import { SortField, SortDirection } from '../types/evalTasks'
import { sortTasks } from '../utils/evalTasks'

export const useEvalTasks = (repo: any) => {
  const [tasks, setTasks] = useState<EvalTask[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeSortField, setActiveSortField] = useState<SortField>('created_at')
  const [activeSortDirection, setActiveSortDirection] = useState<SortDirection>('desc')
  const [completedSortField, setCompletedSortField] = useState<SortField>('created_at')
  const [completedSortDirection, setCompletedSortDirection] = useState<SortDirection>('desc')
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set())

  const loadTasks = useCallback(
    async (search?: string) => {
      setLoading(true)
      try {
        const tasks = await repo.getEvalTasks(search)
        setTasks(tasks)
        setError(null)
      } catch (err: any) {
        console.error('Failed to load tasks:', err)
        setError(err.message || 'Failed to load tasks')
      } finally {
        setLoading(false)
      }
    },
    [repo]
  )

  // Initial load
  useEffect(() => {
    loadTasks()
  }, [loadTasks])

  const handleSort = useCallback((field: SortField, isActive: boolean) => {
    if (isActive) {
      setActiveSortField(field)
      setActiveSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setCompletedSortField(field)
      setCompletedSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    }
  }, [])

  const toggleRowExpansion = useCallback((taskId: string) => {
    setExpandedRows((prev) => {
      const newExpanded = new Set(prev)
      if (newExpanded.has(taskId)) {
        newExpanded.delete(taskId)
      } else {
        newExpanded.add(taskId)
      }
      return newExpanded
    })
  }, [])

  // Filter and sort tasks
  const activeTasks = tasks.filter((t) => t.status === 'unprocessed')
  const historyTasks = tasks.filter((t) => t.status !== 'unprocessed')
  const sortedActiveTasks = sortTasks(activeTasks, activeSortField, activeSortDirection)
  const sortedHistoryTasks = sortTasks(historyTasks, completedSortField, completedSortDirection)

  return {
    tasks,
    loading,
    error,
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
  }
}
