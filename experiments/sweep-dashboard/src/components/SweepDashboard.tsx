import { useState, useEffect } from 'react'
import { SweepData } from '../types'
import { SkyJobsMonitor } from './SkyJobsMonitor'
import { ActiveRunsTable } from './ActiveRunsTable'
import { ModelConfidence } from './ModelConfidence'
import { SweepMetrics } from './SweepMetrics'
import { SweepCharts } from './SweepCharts'
import { SweepFilters } from './SweepFilters'
import { fetchSweepData } from '../api/sweepApi'

interface SweepDashboardProps {
  sweepData: SweepData
  sweepName: string
}

export function SweepDashboard({ sweepData, sweepName }: SweepDashboardProps) {
  const [currentData, setCurrentData] = useState<SweepData>(sweepData)
  const [filteredData, setFilteredData] = useState<SweepData>(sweepData)
  const [isLoadingData, setIsLoadingData] = useState(false)
  const [scoreRange, setScoreRange] = useState<[number, number]>([
    Math.min(...sweepData.runs.map(r => r.score)),
    Math.max(...sweepData.runs.map(r => r.score))
  ])
  const [costRange, setCostRange] = useState<[number, number]>([
    Math.min(...sweepData.runs.map(r => r.cost)),
    Math.max(...sweepData.runs.map(r => r.cost))
  ])
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Auto-refresh sweep data every 30 seconds
  useEffect(() => {
    if (!autoRefresh) return

    const interval = setInterval(async () => {
      try {
        setIsLoadingData(true)
        const freshData = await fetchSweepData(sweepName)
        setCurrentData(freshData)
        
        // Update filtered data while preserving filters
        const filtered = {
          ...freshData,
          runs: freshData.runs.filter(run => 
            run.score >= scoreRange[0] && 
            run.score <= scoreRange[1] &&
            run.cost >= costRange[0] && 
            run.cost <= costRange[1]
          )
        }
        
        setFilteredData({
          ...filtered,
          totalRuns: filtered.runs.length,
          bestScore: filtered.runs.length > 0 ? Math.max(...filtered.runs.map(r => r.score)) : 0,
          totalCost: filtered.runs.reduce((sum, r) => sum + r.cost, 0),
          avgRuntime: filtered.runs.length > 0 ? filtered.runs.reduce((sum, r) => sum + r.runtime, 0) / filtered.runs.length : 0,
          activeRuns: freshData.activeRuns // Always show all active runs
        })
      } catch (error) {
        console.error('Failed to refresh sweep data:', error)
      } finally {
        setIsLoadingData(false)
      }
    }, 30000)

    return () => clearInterval(interval)
  }, [autoRefresh, sweepName, scoreRange, costRange])

  // Update when sweepData prop changes
  useEffect(() => {
    setCurrentData(sweepData)
    setFilteredData(sweepData)
  }, [sweepData])

  const handleFilterChange = (newScoreRange: [number, number], newCostRange: [number, number]) => {
    setScoreRange(newScoreRange)
    setCostRange(newCostRange)
    
    const filtered = {
      ...currentData,
      runs: currentData.runs.filter(run => 
        run.score >= newScoreRange[0] && 
        run.score <= newScoreRange[1] &&
        run.cost >= newCostRange[0] && 
        run.cost <= newCostRange[1]
      )
    }
    
    setFilteredData({
      ...filtered,
      totalRuns: filtered.runs.length,
      bestScore: filtered.runs.length > 0 ? Math.max(...filtered.runs.map(r => r.score)) : 0,
      totalCost: filtered.runs.reduce((sum, r) => sum + r.cost, 0),
      avgRuntime: filtered.runs.length > 0 ? filtered.runs.reduce((sum, r) => sum + r.runtime, 0) / filtered.runs.length : 0,
      activeRuns: currentData.activeRuns // Always show all active runs
    })
  }

  const handleReset = () => {
    const originalScoreRange: [number, number] = [
      Math.min(...currentData.runs.map(r => r.score)),
      Math.max(...currentData.runs.map(r => r.score))
    ]
    const originalCostRange: [number, number] = [
      Math.min(...currentData.runs.map(r => r.cost)),
      Math.max(...currentData.runs.map(r => r.cost))
    ]
    
    setScoreRange(originalScoreRange)
    setCostRange(originalCostRange)
    setFilteredData(currentData)
  }

  return (
    <div>
      {/* Auto-refresh toggle */}
      <div style={{ marginBottom: '20px', display: 'flex', justifyContent: 'flex-end' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
          />
          Auto-refresh data (30s)
        </label>
      </div>

      {/* Sky Jobs Monitor */}
      <SkyJobsMonitor sweepName={sweepName} />

      {/* Active Training Runs */}
      <ActiveRunsTable activeRuns={filteredData.activeRuns || []} isLoading={isLoadingData} />

      {/* Model Confidence Dashboard */}
      <ModelConfidence sweepName={sweepName} />

      {/* Summary Metrics */}
      <SweepMetrics data={filteredData} />

      {/* Filters */}
      <SweepFilters
        scoreRange={scoreRange}
        costRange={costRange}
        maxScore={currentData.runs.length > 0 ? Math.max(...currentData.runs.map(r => r.score)) : 1}
        minScore={currentData.runs.length > 0 ? Math.min(...currentData.runs.map(r => r.score)) : 0}
        maxCost={currentData.runs.length > 0 ? Math.max(...currentData.runs.map(r => r.cost)) : 100}
        minCost={currentData.runs.length > 0 ? Math.min(...currentData.runs.map(r => r.cost)) : 0}
        onFilterChange={handleFilterChange}
        onReset={handleReset}
      />

      {/* Visualizations */}
      <SweepCharts data={filteredData} />
    </div>
  )
}