import React, { useState } from "react";
import { EvalCategory, EvalMetadata } from "./types";

interface EvalCategoryViewProps {
  evalStructure: EvalCategory[];
  selectedEvals: string[];
  onEvalToggle: (evalName: string) => void;
  showPrerequisites: boolean;
}

export const EvalCategoryView: React.FC<EvalCategoryViewProps> = ({
  evalStructure,
  selectedEvals,
  onEvalToggle,
  showPrerequisites
}) => {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(evalStructure.map(category => category.name))
  );

  const toggleCategory = (categoryName: string) => {
    setExpandedCategories(prev => {
      const newSet = new Set(prev);
      if (newSet.has(categoryName)) {
        newSet.delete(categoryName);
      } else {
        newSet.add(categoryName);
      }
      return newSet;
    });
  };

  // Removed difficulty badge rendering - API doesn't provide difficulty data

  const renderPrerequisites = (prerequisites: string[]) => {
    // API doesn't provide prerequisites, so don't show empty data
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

  // Removed agent requirements rendering - API doesn't provide this data

  const renderPolicyStatus = (metadata: EvalMetadata) => {
    // Don't show synthetic status indicators - keep it clean
    return null;
  };

  return (
    <div className="eval-category-view">
      {evalStructure.map((category) => (
        <div key={category.name} className="category-section">
          <div 
            className="category-header" 
            onClick={() => toggleCategory(category.name)}
          >
            <span className="expand-icon">
              {expandedCategories.has(category.name) ? "▼" : "▶"}
            </span>
            <h4>{category.name}</h4>
            <span className="eval-count">
              ({category.children.length} evals)
            </span>
          </div>

          {expandedCategories.has(category.name) && (
            <div className="eval-list">
              {category.children.map((evalNode) => {
                if (!evalNode.eval_metadata) return null;
                
                const metadata = evalNode.eval_metadata;
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
                      <span className="eval-name">{evalNode.name}</span>
                      {renderPolicyStatus(metadata)}
                    </div>
                    
                    {metadata.description && (
                      <div className="eval-description">
                        {metadata.description}
                      </div>
                    )}
                    
                    {renderPrerequisites(metadata.prerequisites)}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};