import { useEffect, useState } from "react";
import { GroupHeatmapMetric, HeatmapData, Repo } from "./repo";
import { MapViewer } from "./MapViewer";
import { Heatmap } from "./Heatmap";

// CSS for dashboard
const DASHBOARD_CSS = `
/* Tab styles */
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

.suite-tab:first-child {
  margin-left: 0;
}

.suite-tab:last-child {
  margin-right: 0;
}
`;

interface DashboardProps {
  repo: Repo;
}

export function Dashboard({ repo }: DashboardProps) {
  // Data state
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null);
  const [metrics, setMetrics] = useState<string[]>([]);
  const [suites, setSuites] = useState<string[]>([]);

  // UI state
  const [selectedMetric, setSelectedMetric] = useState<string>("reward");
  const [selectedSuite, setSelectedSuite] = useState<string>("navigation");
  const [isViewLocked, setIsViewLocked] = useState(false);
  const [selectedCell, setSelectedCell] = useState<{
    policyUri: string;
    evalName: string;
  } | null>(null);
  const [availableGroupMetrics, setAvailableGroupMetrics] = useState<string[]>(
    []
  );
  const [selectedGroupMetric, setSelectedGroupMetric] = useState<string>("");
  const [numPoliciesToShow, setNumPoliciesToShow] = useState(20);

  const parseGroupMetric = (label: string): GroupHeatmapMetric => {
    if (label.includes(" - ")) {
      const [group1, group2] = label.split(" - ");
      return { group_1: group1, group_2: group2 };
    } else {
      return label;
    }
  };

  useEffect(() => {
    const loadData = async () => {
      const suitesData = await repo.getSuites();
      setSuites(suitesData);
      setSelectedSuite(suitesData[0]);
    };

    loadData();
  }, []);

  useEffect(() => {
    const loadData = async () => {
      const [metricsData, groupIdsData] = await Promise.all([
        repo.getMetrics(selectedSuite),
        repo.getGroupIds(selectedSuite),
      ]);
      setMetrics(metricsData);

      const groupDiffs: string[] = [];
      for (const groupId1 of groupIdsData) {
        for (const groupId2 of groupIdsData) {
          if (groupId1 !== groupId2) {
            groupDiffs.push(`${groupId1} - ${groupId2}`);
          }
        }
      }

      const groupMetrics: string[] = ["", ...groupIdsData, ...groupDiffs];
      setAvailableGroupMetrics(groupMetrics);
    };

    loadData();
  }, [selectedSuite]);

  useEffect(() => {
    const loadData = async () => {
      const heatmapData = await repo.getHeatmapData(
        selectedMetric,
        selectedSuite,
        parseGroupMetric(selectedGroupMetric)
      );
      setHeatmapData(heatmapData);
    };

    loadData();
  }, [selectedSuite, selectedMetric, selectedGroupMetric]);

  if (!heatmapData) {
    return <div>Loading...</div>;
  }

  // Component functions

  const setSelectedCellIfNotLocked = (cell: {
    policyUri: string;
    evalName: string;
  }) => {
    if (!isViewLocked) {
      setSelectedCell(cell);
    }
  };

  const openReplayUrl = (policyUri: string, evalName: string) => {
    const evalData = heatmapData?.cells[policyUri]?.[evalName];
    if (!evalData?.replayUrl) return;

    const replay_url_prefix = "https://metta-ai.github.io/metta/?replayUrl=";
    window.open(replay_url_prefix + evalData.replayUrl, "_blank");
  };

  const toggleLock = () => {
    setIsViewLocked(!isViewLocked);
  };

  const handleReplayClick = () => {
    if (selectedCell) {
      openReplayUrl(selectedCell.policyUri, selectedCell.evalName);
    }
  };

  const selectedCellData = selectedCell
    ? heatmapData?.cells[selectedCell.policyUri]?.[selectedCell.evalName]
    : null;
  const selectedEval = selectedCellData?.evalName ?? null;
  const selectedReplayUrl = selectedCellData?.replayUrl ?? null;

  return (
    <div
      style={{
        fontFamily: "Arial, sans-serif",
        margin: 0,
        padding: "20px",
        background: "#f8f9fa",
      }}
    >
      <style>{DASHBOARD_CSS}</style>
      <div
        style={{
          maxWidth: "1200px",
          margin: "0 auto",
          background: "#fff",
          padding: "20px",
          borderRadius: "5px",
          boxShadow: "0 2px 4px rgba(0,0,0,.1)",
        }}
      >
        <h1
          style={{
            color: "#333",
            borderBottom: "1px solid #ddd",
            paddingBottom: "10px",
            marginBottom: "20px",
          }}
        >
          Policy Evaluation Dashboard
        </h1>

        <div className="suite-tabs">
          <div
            style={{ fontSize: "18px", marginTop: "5px", marginRight: "10px" }}
          >
            Eval Suite:
          </div>
          {suites.map((suite) => (
            <button
              key={suite}
              className={`suite-tab ${selectedSuite === suite ? "active" : ""}`}
              onClick={() => setSelectedSuite(suite)}
            >
              {suite}
            </button>
          ))}
        </div>
        {heatmapData && (
          <Heatmap
            data={heatmapData}
            selectedMetric={selectedMetric}
            setSelectedCell={setSelectedCellIfNotLocked}
            openReplayUrl={openReplayUrl}
            numPoliciesToShow={numPoliciesToShow}
          />
        )}

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            marginTop: "20px",
            marginBottom: "30px",
            gap: "12px",
          }}
        >
          <div style={{ color: "#666", fontSize: "14px" }}>Heatmap Metric</div>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            style={{
              padding: "8px 12px",
              borderRadius: "4px",
              border: "1px solid #ddd",
              fontSize: "14px",
              minWidth: "200px",
              backgroundColor: "#fff",
              cursor: "pointer",
            }}
          >
            {metrics.map((metric) => (
              <option key={metric} value={metric}>
                {metric}
              </option>
            ))}
          </select>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            marginTop: "20px",
            marginBottom: "30px",
            gap: "12px",
          }}
        >
          <div style={{ color: "#666", fontSize: "14px" }}>Group Metric</div>
          <select
            value={selectedGroupMetric}
            onChange={(e) => setSelectedGroupMetric(e.target.value)}
            style={{
              padding: "8px 12px",
              borderRadius: "4px",
              border: "1px solid #ddd",
              fontSize: "14px",
              minWidth: "200px",
              backgroundColor: "#fff",
              cursor: "pointer",
            }}
          >
            {availableGroupMetrics.map((groupMetric) => (
              <option key={groupMetric} value={groupMetric}>
                {groupMetric === "" ? "Total" : groupMetric}
              </option>
            ))}
          </select>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            marginTop: "20px",
            marginBottom: "30px",
            gap: "12px",
          }}
        >
          <div style={{ color: "#666", fontSize: "14px" }}>
            Number of policies to show:
          </div>
          <input
            type="number"
            value={numPoliciesToShow}
            onChange={(e) => setNumPoliciesToShow(parseInt(e.target.value))}
            style={{
              padding: "8px 12px",
              borderRadius: "4px",
              border: "1px solid #ddd",
              fontSize: "14px",
            }}
          />
        </div>

        <MapViewer
          selectedEval={selectedEval}
          isViewLocked={isViewLocked}
          selectedReplayUrl={selectedReplayUrl}
          onToggleLock={toggleLock}
          onReplayClick={handleReplayClick}
        />
      </div>
    </div>
  );
}
