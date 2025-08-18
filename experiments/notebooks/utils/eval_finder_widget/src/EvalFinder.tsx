import React, { useState, useEffect, useMemo } from "react";
import { EvalCategoryView } from "./EvalCategoryView";
import { EvalListView } from "./EvalListView";
import { FilterPanel } from "./FilterPanel";
import { SearchBar } from "./SearchBar";
import { EvalMetadata, EvalCategory } from "./types";

interface EvalFinderProps {
  evalData: any;
  selectedEvals: string[];
  categoryFilter: string[];
  viewMode: "tree" | "list" | "category";
  searchTerm: string;
  showPrerequisites: boolean;
  onSelectionChange: (selected: string[]) => void;
  onFilterChange: (filters: any) => void;
}

export const EvalFinder: React.FC<EvalFinderProps> = ({ 
  evalData,
  selectedEvals,
  categoryFilter,
  viewMode,
  searchTerm,
  showPrerequisites,
  onSelectionChange,
  onFilterChange
}) => {
  // Local UI state
  const [localSearchTerm, setLocalSearchTerm] = useState<string>(searchTerm);

  // Use localSearchTerm for immediate UI updates, but sync with parent on change
  useEffect(() => {
    setLocalSearchTerm(searchTerm);
  }, [searchTerm]);

  // Filter evaluations based on current filters
  const filteredEvaluations = useMemo(() => {
    if (!evalData?.evaluations) {
      return [];
    }

    const filtered = evalData.evaluations.filter((evaluation: EvalMetadata) => {
      // Search filter (use local search term for immediate feedback)
      if (localSearchTerm) {
        const searchLower = localSearchTerm.toLowerCase();
        const matchesName = evaluation.name.toLowerCase().includes(searchLower);
        const matchesDescription = evaluation.description?.toLowerCase().includes(searchLower) || false;
        const matchesTags = evaluation.tags?.some(tag => tag.toLowerCase().includes(searchLower)) || false;

        if (!matchesName && !matchesDescription && !matchesTags) {
          return false;
        }
      }

      // Category filter
      if (categoryFilter.length > 0 && !categoryFilter.includes(evaluation.category)) {
        return false;
      }

      return true;
    });

    return filtered;
  }, [evalData, localSearchTerm, categoryFilter]);

  // Filter eval structure based on filtered evaluations
  const filteredEvalStructure = useMemo(() => {
    if (!evalData?.categories) return [];

    const filteredEvalNames = new Set(filteredEvaluations.map((e: EvalMetadata) => e.name));

    return evalData.categories.map((category: EvalCategory) => ({
      ...category,
      children: category.children.filter((evalNode: any) =>
        evalNode.eval_metadata && filteredEvalNames.has(evalNode.eval_metadata.name)
      )
    })).filter((category: EvalCategory) => category.children.length > 0);
  }, [evalData, filteredEvaluations]);

  const handleEvalToggle = (evalName: string) => {
    const newSelection = selectedEvals.includes(evalName)
      ? selectedEvals.filter(name => name !== evalName)
      : [...selectedEvals, evalName];
    onSelectionChange(newSelection);
  };

  const handleSelectAll = () => {
    const allEvalNames = filteredEvaluations.map((e: EvalMetadata) => e.name);
    onSelectionChange(allEvalNames);
  };

  const handleClearAll = () => {
    onSelectionChange([]);
  };

  const handleFilterChange = (filters: {
    category?: string[];
  }) => {
    onFilterChange({
      categoryFilter: filters.category,
      searchTerm: localSearchTerm,
      viewMode,
      showPrerequisites
    });
  };

  const handleSearchChange = (term: string) => {
    setLocalSearchTerm(term);
    onFilterChange({
      categoryFilter,
      searchTerm: term,
      viewMode,
      showPrerequisites
    });
  };

  if (!evalData) {
    return (
      <div className="eval-finder-container">
        <div className="loading-message">
          🔍 Loading evaluations...
        </div>
      </div>
    );
  }

  return (
    <div className="eval-finder-container">
      <div className="eval-finder-header">
        <h3>🎯 Evaluation Finder</h3>
        <div className="stats">
          {selectedEvals.length} of {filteredEvaluations.length} selected
        </div>
      </div>

      <SearchBar
        searchTerm={localSearchTerm}
        onSearchChange={handleSearchChange}
      />

      <FilterPanel
        categoryFilter={categoryFilter}
        availableCategories={evalData.categories?.map((category: EvalCategory) => category.name) || []}
        onFilterChange={handleFilterChange}
      />

      <div className="view-controls">
        <div className="view-mode-selector">
          <label>View:</label>
          <select
            value={viewMode}
            onChange={(e) => onFilterChange({
              categoryFilter,
              searchTerm: localSearchTerm,
              viewMode: e.target.value as "tree" | "list" | "category",
              showPrerequisites
            })}
          >
            <option value="tree">Tree</option>
            <option value="list">List</option>
            <option value="category">By Category</option>
          </select>
        </div>

        <div className="selection-controls">
          <button
            onClick={handleSelectAll}
            className="btn-select-all"
            disabled={selectedEvals.length === filteredEvaluations.length}
            title={`Select all ${filteredEvaluations.length} evaluations`}
          >
            📋 Select All ({filteredEvaluations.length})
          </button>
          <button
            onClick={handleClearAll}
            className="btn-clear-all"
            disabled={selectedEvals.length === 0}
            title="Clear all selections"
          >
            🗑️ Clear ({selectedEvals.length})
          </button>
        </div>
      </div>

      <div className="eval-content">
        {filteredEvaluations.length === 0 ? (
          <div className="no-evals-message">
            <h4>No evaluations found</h4>
            {evalData?.evaluations?.length > 0 ? (
              <p>No evaluations match your current filters. Try adjusting your search or category filters.</p>
            ) : (
              <div>
                <p>No evaluation data available.</p>
              </div>
            )}
          </div>
        ) : (
          <>
            {viewMode === "tree" ? (
              <EvalCategoryView
                evalStructure={filteredEvalStructure}
                selectedEvals={selectedEvals}
                onEvalToggle={handleEvalToggle}
                showPrerequisites={showPrerequisites}
              />
            ) : (
              <EvalListView
                evaluations={filteredEvaluations}
                selectedEvals={selectedEvals}
                onEvalToggle={handleEvalToggle}
                viewMode={viewMode}
                showPrerequisites={showPrerequisites}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
};
