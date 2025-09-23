import { useState, useEffect, useCallback, useRef } from 'react'
import { EvalTask, Repo } from '../repo'
import { SortField, SortDirection } from '../types/evalTasks'
import { sortTasks } from '../utils/evalTasksUtils'

export const useEvalTasks = (repo: Repo) => {
  const [tasks, setTasks] = useState<EvalTask[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeSortField, setActiveSortField] = useState<SortField>('created_at')
  const [activeSortDirection, setActiveSortDirection] = useState<SortDirection>('desc')
  const [completedSortField, setCompletedSortField] = useState<SortField>('created_at')
  const [completedSortDirection, setCompletedSortDirection] = useState<SortDirection>('desc')
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set())

  // Refs for debouncing and cancellation
  const abortControllerRef = useRef<AbortController | null>(null)
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null)

  const loadTasks = useCallback(
    async (search?: string) => {
      // Cancel previous request if it exists
      if (abortControllerRef.current) {
        console.log('Aborting previous request')
        abortControllerRef.current.abort()
      }

      // Create new abort controller for this request
      const abortController = new AbortController()
      abortControllerRef.current = abortController

      setLoading(true)
      try {
        const tasks = await repo.getEvalTasks(search)

        // Only update state if the request wasn't aborted
        if (!abortController.signal.aborted) {
          setTasks(tasks)
          setError(null)
        }
      } catch (err: any) {
        // Ignore abort errors (they're expected when cancelling)
        if (err.name === 'AbortError') {
          return
        }

        console.error('Failed to load tasks:', err)
        if (!abortController.signal.aborted) {
          setError(err.message || 'Failed to load tasks')
        }
      } finally {
        // Only update loading state if not aborted
        if (!abortController.signal.aborted) {
          setLoading(false)
        }
      }
    },
    [repo]
  )

  // Debounced version of loadTasks
  const debouncedLoadTasks = useCallback(
    (search?: string, delay: number = 300) => {
      // Clear previous debounce timer
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }

      // Set new debounce timer
      debounceTimerRef.current = setTimeout(() => {
        loadTasks(search)
      }, delay)
    },
    [loadTasks]
  )

  // Initial load
  useEffect(() => {
    loadTasks()
  }, [loadTasks])

  // Cleanup: abort any pending requests and clear timers on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        console.log('Aborting request')
        abortControllerRef.current.abort()
      }
      if (debounceTimerRef.current) {
        console.log('Clearing debounce timer')
        clearTimeout(debounceTimerRef.current)
      }
    }
  }, [])

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
    loadTasks: debouncedLoadTasks,
    handleSort,
    toggleRowExpansion,
    activeTasks,
    historyTasks,
    sortedActiveTasks,
    sortedHistoryTasks,
  }
}
