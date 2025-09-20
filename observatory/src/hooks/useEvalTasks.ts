import { useState, useEffect } from 'react'
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

    const loadTasks = async (search?: string) => {
        try {
            const tasks = await repo.getEvalTasks(search)
            setTasks(tasks)
        } catch (err: any) {
            console.error('Failed to refresh tasks:', err)
        }
    }

    // Set up auto-refresh for tasks
    useEffect(() => {
        loadTasks()
        const interval = setInterval(() => loadTasks(), 5000) // Refresh every 5 seconds

        return () => {
            clearInterval(interval)
        }
    }, [])

    const handleSort = (field: SortField, isActive: boolean) => {
        if (isActive) {
            setActiveSortField(field)
            setActiveSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))
        } else {
            setCompletedSortField(field)
            setCompletedSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))
        }
    }

    const toggleRowExpansion = (taskId: string) => {
        const newExpanded = new Set(expandedRows)
        if (newExpanded.has(taskId)) {
            newExpanded.delete(taskId)
        } else {
            newExpanded.add(taskId)
        }
        setExpandedRows(newExpanded)
    }

    const activeTasks = tasks.filter((t) => t.status === 'unprocessed')
    const historyTasks = tasks.filter((t) => t.status !== 'unprocessed')

    const sortedActiveTasks = sortTasks(activeTasks, activeSortField, activeSortDirection)
    const sortedHistoryTasks = sortTasks(historyTasks, completedSortField, completedSortDirection)

    return {
        tasks,
        setTasks,
        loading,
        setLoading,
        error,
        setError,
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
