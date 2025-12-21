import React from 'react'
import { useModelState, createRender } from '@anywidget/react'
import { EvalFinder } from './EvalFinder'

function EvalFinderWidget() {
  // Use anywidget React bridge hooks for state management
  const [evalData] = useModelState<any>('eval_data')
  const [selectedEvals, setSelectedEvals] = useModelState<string[]>('selected_evals')
  const [categoryFilter, setCategoryFilter] = useModelState<string[]>('category_filter')
  const [viewMode, setViewMode] = useModelState<string>('view_mode')
  const [searchTerm, setSearchTerm] = useModelState<string>('search_term')
  const [showPrerequisites, setShowPrerequisites] = useModelState<boolean>('show_prerequisites')

  // Event traits for communication with Python
  const [, setSelectionChanged] = useModelState<any>('selection_changed')
  const [, setFilterChanged] = useModelState<any>('filter_changed')

  return (
    <EvalFinder
      evalData={evalData}
      selectedEvals={selectedEvals || []}
      categoryFilter={categoryFilter || []}
      viewMode={(viewMode || 'category') as 'tree' | 'list' | 'category'}
      searchTerm={searchTerm || ''}
      showPrerequisites={showPrerequisites !== false}
      onSelectionChange={(selected: string[]) => {
        setSelectedEvals(selected)
        setSelectionChanged({
          selected_evals: selected,
          action: 'updated',
          timestamp: Date.now(),
        })
      }}
      onFilterChange={(filters: any) => {
        setCategoryFilter(filters.categoryFilter || [])
        setSearchTerm(filters.searchTerm || '')
        setViewMode(filters.viewMode || 'category')
        setShowPrerequisites(filters.showPrerequisites !== false)
        setFilterChanged({
          ...filters,
          timestamp: Date.now(),
        })
      }}
    />
  )
}

export default {
  render: createRender(EvalFinderWidget),
}
