import { useState, useEffect, useCallback, useRef } from 'react'

export const useSearch = (loadTasks: (search?: string) => Promise<void>) => {
    const [searchText, setSearchText] = useState('')
    const [searchLoading, setSearchLoading] = useState(false)
    const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    const abortControllerRef = useRef<AbortController | null>(null)
    const currentSearchRef = useRef<string>('')

    // Debounced search function
    const debouncedSearch = useCallback(
        (searchValue: string) => {
            // Clear existing timeout
            if (debounceTimeoutRef.current) {
                clearTimeout(debounceTimeoutRef.current)
            }

            // Cancel any ongoing request
            if (abortControllerRef.current) {
                abortControllerRef.current.abort()
            }

            // Store current search to avoid race conditions
            currentSearchRef.current = searchValue

            // Set loading state immediately for better UX
            setSearchLoading(true)

            // Set new timeout
            debounceTimeoutRef.current = setTimeout(async () => {
                // Double-check this is still the current search
                if (currentSearchRef.current !== searchValue) {
                    setSearchLoading(false)
                    return
                }

                // Create new abort controller for this request
                abortControllerRef.current = new AbortController()

                try {
                    const trimmedValue = searchValue.trim()
                    await loadTasks(trimmedValue || undefined)
                } catch (error) {
                    // Only log error if it wasn't aborted
                    if (error.name !== 'AbortError') {
                        console.error('Search failed:', error)
                    }
                } finally {
                    // Only update loading if this is still the current search
                    if (currentSearchRef.current === searchValue) {
                        setSearchLoading(false)
                    }
                }
            }, 250) // Slightly faster - 250ms is still comfortable
        },
        [loadTasks]
    )

    const handleSearchChange = useCallback(
        (value: string) => {
            setSearchText(value)
            debouncedSearch(value)
        },
        [debouncedSearch]
    )

    // Handle immediate search for empty string (clear search)
    useEffect(() => {
        if (searchText === '') {
            // Clear any pending debounced search
            if (debounceTimeoutRef.current) {
                clearTimeout(debounceTimeoutRef.current)
            }

            // Cancel any ongoing request
            if (abortControllerRef.current) {
                abortControllerRef.current.abort()
            }

            // Update current search reference
            currentSearchRef.current = ''

            // Immediately search with empty string to show all results
            setSearchLoading(true)
            loadTasks().finally(() => {
                // Only update loading if search is still empty
                if (currentSearchRef.current === '') {
                    setSearchLoading(false)
                }
            })
        }
    }, [searchText, loadTasks])

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (debounceTimeoutRef.current) {
                clearTimeout(debounceTimeoutRef.current)
            }
            if (abortControllerRef.current) {
                abortControllerRef.current.abort()
            }
        }
    }, [])

    // Optional: Add a method to manually trigger search (useful for refresh)
    const triggerSearch = useCallback(() => {
        debouncedSearch(searchText)
    }, [debouncedSearch, searchText])

    return {
        searchText,
        searchLoading,
        handleSearchChange,
        triggerSearch, // Expose for manual refresh if needed
    }
}
