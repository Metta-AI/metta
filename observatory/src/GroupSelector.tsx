import { useEffect, useState } from 'react'
import type { GroupHeatmapMetric, Repo } from './repo'

const GROUP_SELECTOR_CSS = `
.group-selector {
  padding: 8px 12px;
  border-radius: 4px;
  border: 1px solid #ddd;
  font-size: 14px;
  flex: 1;
  background-color: #fff;
  cursor: pointer;
}

.group-selector:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
`

export const parseGroupMetric = (label: string): GroupHeatmapMetric => {
  if (label.includes(' - ')) {
    const [group1, group2] = label.split(' - ')
    return { group_1: group1, group_2: group2 }
  }
  return label
}

const generateGroupMetrics = (groupIds: Array<string>): Array<string> => {
  const groupDiffs: Array<string> = []
  for (const groupId1 of groupIds) {
    for (const groupId2 of groupIds) {
      if (groupId1 !== groupId2) {
        groupDiffs.push(`${groupId1} - ${groupId2}`)
      }
    }
  }
  return ['', ...groupIds, ...groupDiffs]
}

interface GroupSelectorProps {
  repo: Repo
  selectedSuite: string
  selectedGroupMetric: string
  onGroupMetricChange: (groupMetric: string) => void
}

export function GroupSelector({ repo, selectedSuite, selectedGroupMetric, onGroupMetricChange }: GroupSelectorProps) {
  const [availableGroupMetrics, setAvailableGroupMetrics] = useState<Array<string>>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const loadGroupMetrics = async () => {
      if (!selectedSuite) {
        return
      }

      setLoading(true)
      try {
        const groupIds = await repo.getGroupIds(selectedSuite)
        const groupMetrics = generateGroupMetrics(groupIds)
        setAvailableGroupMetrics(groupMetrics)
      } catch (err) {
        console.error('Failed to load group metrics:', err)
        setAvailableGroupMetrics([''])
      } finally {
        setLoading(false)
      }
    }

    loadGroupMetrics()
  }, [selectedSuite, repo])

  return (
    <>
      <style>{GROUP_SELECTOR_CSS}</style>
      <select
        value={selectedGroupMetric}
        onChange={(e) => onGroupMetricChange(e.target.value)}
        className="group-selector"
        disabled={loading}
      >
        {availableGroupMetrics.map((groupMetric) => (
          <option key={groupMetric} value={groupMetric}>
            {groupMetric === '' ? 'Total' : groupMetric}
          </option>
        ))}
      </select>
    </>
  )
}
