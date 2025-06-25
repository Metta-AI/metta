import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { GroupHeatmapMetric, HeatmapData, Repo, SavedDashboard, SavedDashboardCreate } from "./repo";
import { MapViewer } from "./MapViewer";
import { Heatmap } from "./Heatmap";
import { SaveDashboardModal } from "./SaveDashboardModal";

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

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s ease;
}

.btn-primary {
  background: #007bff;
  color: #fff;
}

.btn-primary:hover {
  background: #0056b3;
}

.btn-secondary {
  background: #6c757d;
  color: #fff;
}

.btn-secondary:hover {
  background: #545b62;
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

  // Save dashboard state
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [savedId, setSavedId] = useState<string | null>(null);
  const [savedDashboard, setSavedDashboard] = useState<SavedDashboard | null>(null);

  const location = useLocation();

  const parseGroupMetric = (label: string): GroupHeatmapMetric => {
    if (label.includes(" - ")) {
      const [group1, group2] = label.split(" - ");
      return { group_1: group1, group_2: group2 };
    } else {
      return label;
    }
  };

  // Initialize data and load saved dashboard if provided
  useEffect(() => {
    const initializeData = async () => {
      const urlParams = new URLSearchParams(location.search);
      const savedIdParam = urlParams.get('saved_id');

      // Load suites first
      const suitesData = await repo.getSuites();
      setSuites(suitesData);

      if (savedIdParam) {
        // Load saved dashboard
        try {
          const dashboard = await repo.getSavedDashboard(savedIdParam);
          const state = dashboard.dashboard_state;
          setSelectedSuite(state.suite || suitesData[0]);
          setSelectedMetric(state.metric || "reward");
          setSelectedGroupMetric(state.group_metric || "");
          setNumPoliciesToShow(state.num_policies_to_show || 20);
          setSavedId(savedIdParam);
          setSavedDashboard(dashboard);
        } catch (err) {
          console.error("Failed to load shared dashboard:", err);
          // Fallback to first suite if saved dashboard fails
          setSelectedSuite(suitesData[0]);
        }
      } else {
        // No saved dashboard, use first suite
        setSelectedSuite(suitesData[0]);
      }
    };

    initializeData();
  }, [location.search, repo]);

  // Load metrics and group metrics when suite changes
  useEffect(() => {
    const loadSuiteData = async () => {
      if (!selectedSuite) return;

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

    loadSuiteData();
  }, [selectedSuite, repo]);

  // Load heatmap data when suite, metric, or group metric changes
  useEffect(() => {
    const loadHeatmapData = async () => {
      if (!selectedSuite || !selectedMetric) return;

      const heatmapData = await repo.getHeatmapData(
        selectedMetric,
        selectedSuite,
        parseGroupMetric(selectedGroupMetric)
      );
      setHeatmapData(heatmapData);
    };

    loadHeatmapData();
  }, [selectedSuite, selectedMetric, selectedGroupMetric, repo]);

  const handleSaveDashboard = async (dashboardData: SavedDashboardCreate) => {
    try {
      const fullDashboardData: SavedDashboardCreate = {
        ...dashboardData,
        dashboard_state: {
          suite: selectedSuite,
          metric: selectedMetric,
          group_metric: selectedGroupMetric,
          num_policies_to_show: numPoliciesToShow,
        },
      };

      if (savedId) {
        // Update existing dashboard
        const updatedDashboard = await repo.updateSavedDashboard(savedId, fullDashboardData);
        setSavedDashboard(updatedDashboard);
      } else {
        // Create new dashboard
        const newDashboard = await repo.createSavedDashboard(fullDashboardData);
        setSavedId(newDashboard.id);
        setSavedDashboard(newDashboard);
      }
    } catch (err: any) {
      throw new Error(err.message || "Failed to save dashboard");
    }
  };

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
        padding: "20px",
        background: "#f8f9fa",
        minHeight: "calc(100vh - 60px)",
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
        {savedDashboard && (
          <div style={{
            textAlign: "center",
            marginBottom: "20px",
            paddingBottom: "20px",
            borderBottom: "1px solid #eee"
          }}>
            <h1 style={{
              margin: 0,
              color: "#333",
              fontSize: "24px",
              fontWeight: "600"
            }}>
              {savedDashboard.name}
            </h1>
            {savedDashboard.description && (
              <p style={{
                margin: "8px 0 0 0",
                color: "#666",
                fontSize: "16px"
              }}>
                {savedDashboard.description}
              </p>
            )}
          </div>
        )}

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
          <div style={{ marginLeft: "auto" }}>
            <button
              className="btn btn-secondary"
              onClick={() => setShowSaveModal(true)}
            >
              {savedId ? "Update Dashboard" : "Save Dashboard"}
            </button>
          </div>
        </div>

        <SaveDashboardModal
          isOpen={showSaveModal}
          onClose={() => setShowSaveModal(false)}
          onSave={handleSaveDashboard}
          initialName={savedDashboard?.name || ""}
          initialDescription={savedDashboard?.description || ""}
          isUpdate={!!savedId}
        />

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
