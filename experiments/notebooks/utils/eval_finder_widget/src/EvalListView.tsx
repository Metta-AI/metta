import React from "react";
import { EvalMetadata } from "./types";

interface EvalListViewProps {
  evaluations: EvalMetadata[];
  selectedEvals: string[];
  onEvalToggle: (evalName: string) => void;
  viewMode: "list" | "category";
  showPrerequisites: boolean;
}

export const EvalListView: React.FC<EvalListViewProps> = ({
  evaluations,
  selectedEvals,
  onEvalToggle,
  viewMode,
  showPrerequisites
}) => {
  const renderDifficultyBadge = (difficulty: string) => {
    const colorMap = {
      easy: "#22c55e",
      medium: "#f59e0b",
      hard: "#ef4444"
    };

    return (
      <span
        className="difficulty-badge"
        style={{ backgroundColor: colorMap[difficulty as keyof typeof colorMap] }}
      >
        {difficulty}
      </span>
    );
  };

  const renderPrerequisites = (prerequisites: string[]) => {
    if (!showPrerequisites || prerequisites.length === 0) return null;

    return (
      <div className="prerequisites">
        <span className="prereq-label">Requires:</span>
        {prerequisites.map((prereq, idx) => (
          <span key={idx} className="prereq-item">
            {prereq.split('/').pop()}
          </span>
        ))}
      </div>
    );
  };

  const renderAgentRequirements = (agentRequirements: string[]) => {
    if (agentRequirements.includes("any")) return null;

    return (
      <div className="agent-requirements">
        {agentRequirements.map((req, idx) => (
          <span key={idx} className="agent-req-badge">
            {req}
          </span>
        ))}
      </div>
    );
  };

  if (viewMode === "category") {
    // Group by category
    const evalsByCategory = evaluations.reduce((acc, eval_meta) => {
      if (!acc[eval_meta.category]) {
        acc[eval_meta.category] = [];
      }
      acc[eval_meta.category].push(eval_meta);
      return acc;
    }, {} as Record<string, EvalMetadata[]>);

    return (
      <div className="eval-list-view category-view">
        {Object.entries(evalsByCategory).map(([category, categoryEvals]) => (
          <div key={category} className="category-section">
            <h4 className="category-title">{category}</h4>
            <div className="eval-grid">
              {categoryEvals.map((metadata) => {
                const isSelected = selectedEvals.includes(metadata.name);

                return (
                  <div
                    key={metadata.name}
                    className={`eval-card ${isSelected ? "selected" : ""}`}
                    onClick={() => onEvalToggle(metadata.name)}
                  >
                    <div className="eval-header">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => onEvalToggle(metadata.name)}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <span className="eval-name">
                        {metadata.name.split('/').pop()}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    );
  }

  // List view
  return (
    <div className="eval-list-view">
      {evaluations.map((metadata) => {
        const isSelected = selectedEvals.includes(metadata.name);

        return (
          <div
            key={metadata.name}
            className={`eval-item ${isSelected ? "selected" : ""}`}
            onClick={() => onEvalToggle(metadata.name)}
          >
            <div className="eval-header">
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => onEvalToggle(metadata.name)}
                onClick={(e) => e.stopPropagation()}
              />
              <span className="eval-name">{metadata.name}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
};
