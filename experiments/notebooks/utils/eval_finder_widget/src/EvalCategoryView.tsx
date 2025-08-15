import React, { useState } from "react";
import { EvalCategory, EvalMetadata } from "./types";

interface EvalCategoryViewProps {
  evalStructure: EvalCategory[];
  selectedEvals: string[];
  onEvalToggle: (evalName: string) => void;
  showPrerequisites: boolean;
}

export const EvalCategoryView = ({
  evalStructure,
  selectedEvals,
  onEvalToggle,
  showPrerequisites
}: EvalCategoryViewProps) => {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(evalStructure.map(category => category.name))
  );

  const toggleCategory = (categoryName: string) => {
    setExpandedCategories((prev: Set<string>) => {
      const newSet = new Set(prev);
      if (newSet.has(categoryName)) {
        newSet.delete(categoryName);
      } else {
        newSet.add(categoryName);
      }
      return newSet;
    });
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
                        onClick={(e: React.MouseEvent<HTMLInputElement>) => e.stopPropagation()}
                      />
                      <span className="eval-name">{evalNode.name}</span>
                    </div>

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
