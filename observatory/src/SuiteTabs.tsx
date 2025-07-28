import type React from 'react'

const SUITE_TABS_CSS = `
.suite-tabs {
  display: flex;
  gap: 2px;
  padding: 4px;
  border-radius: 8px;
  margin-bottom: 20px;
  overflow-x: auto;
  max-width: 1000px;
  margin: 0 auto 20px auto;
}

.suite-tab {
  padding: 8px 16px;
  border: none;
  background: #fff;
  cursor: pointer;
  font-size: 14px;
  color: #666;
  border-radius: 6px;
  white-space: nowrap;
  transition: all 0.2s ease;
}

.suite-tab:hover {
  background: #f8f8f8;
  color: #333;
}

.suite-tab.active {
  background: #007bff;
  color: #fff;
  font-weight: 500;
}
`

interface SuiteTabsProps {
  suites: Array<string>
  selectedSuite: string
  onSuiteChange: (suite: string) => void
  rightContent?: React.ReactNode
}

export function SuiteTabs({ suites, selectedSuite, onSuiteChange, rightContent }: SuiteTabsProps) {
  return (
    <>
      <style>{SUITE_TABS_CSS}</style>
      <div className="suite-tabs">
        <div style={{ fontSize: '18px', marginTop: '5px', marginRight: '10px' }}>Eval Suite:</div>
        {suites.map((suite) => (
          <button
            key={suite}
            className={`suite-tab ${selectedSuite === suite ? 'active' : ''}`}
            onClick={() => onSuiteChange(suite)}
          >
            {suite}
          </button>
        ))}
        {rightContent && <div style={{ marginLeft: 'auto' }}>{rightContent}</div>}
      </div>
    </>
  )
}
