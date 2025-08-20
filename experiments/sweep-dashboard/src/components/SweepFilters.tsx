interface SweepFiltersProps {
  scoreRange: [number, number]
  costRange: [number, number]
  maxScore: number
  minScore: number
  maxCost: number
  minCost: number
  onFilterChange: (scoreRange: [number, number], costRange: [number, number]) => void
  onReset: () => void
}

export function SweepFilters({
  scoreRange,
  costRange,
  maxScore,
  minScore,
  maxCost,
  minCost,
  onFilterChange,
  onReset
}: SweepFiltersProps) {
  return (
    <div className="card">
      <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Filters</h3>
      
      <div style={{ marginBottom: '20px' }}>
        <label className="form-label">
          Score Range: {scoreRange[0].toFixed(3)} - {scoreRange[1].toFixed(3)}
        </label>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <input
            type="range"
            min={minScore}
            max={maxScore}
            step={0.001}
            value={scoreRange[0]}
            onChange={(e) => onFilterChange([parseFloat(e.target.value), scoreRange[1]], costRange)}
            style={{ flex: 1 }}
          />
          <input
            type="range"
            min={minScore}
            max={maxScore}
            step={0.001}
            value={scoreRange[1]}
            onChange={(e) => onFilterChange([scoreRange[0], parseFloat(e.target.value)], costRange)}
            style={{ flex: 1 }}
          />
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <label className="form-label">
          Cost Range: ${costRange[0].toFixed(2)} - ${costRange[1].toFixed(2)}
        </label>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <input
            type="range"
            min={minCost}
            max={maxCost}
            step={0.01}
            value={costRange[0]}
            onChange={(e) => onFilterChange(scoreRange, [parseFloat(e.target.value), costRange[1]])}
            style={{ flex: 1 }}
          />
          <input
            type="range"
            min={minCost}
            max={maxCost}
            step={0.01}
            value={costRange[1]}
            onChange={(e) => onFilterChange(scoreRange, [costRange[0], parseFloat(e.target.value)])}
            style={{ flex: 1 }}
          />
        </div>
      </div>

      <button className="btn btn-secondary btn-sm" onClick={onReset}>
        Reset Filters
      </button>
    </div>
  )
}