import React, { useState, useEffect, useMemo } from "react";
import { EvalCategoryView } from "./EvalCategoryView";
import { EvalListView } from "./EvalListView";
import { FilterPanel } from "./FilterPanel";
import { SearchBar } from "./SearchBar";
import { EvalMetadata, EvalCategory } from "./types";

interface EvalFinderProps {
  model: any;
}

export const EvalFinder: React.FC<EvalFinderProps> = ({ model }) => {
  // Widget state from Python backend
  const [evalData, setEvalData] = useState<any>(null);
  const [selectedEvals, setSelectedEvals] = useState<string[]>([]);
  const [categoryFilter, setCategoryFilter] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<"tree" | "list" | "category">("tree");
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [showPrerequisites, setShowPrerequisites] = useState<boolean>(true);

  // Sync with Python backend
  useEffect(() => {
    const updateFromModel = () => {
      const newEvalData = model.get("eval_data");
      if (newEvalData) {
        //console.log("🔍 EvalFinder: Received eval data:", newEvalData);
        setEvalData(newEvalData);
      }

      const pySelectedEvals = model.get("selected_evals");
      if (pySelectedEvals && JSON.stringify(pySelectedEvals) !== JSON.stringify(selectedEvals)) {
        setSelectedEvals(pySelectedEvals);
      }

      setCategoryFilter(model.get("category_filter") || []);
      setViewMode(model.get("view_mode") || "tree");
      setSearchTerm(model.get("search_term") || "");
      setShowPrerequisites(model.get("show_prerequisites") !== false);
    };

    updateFromModel();
    model.on("change", updateFromModel);

    return () => model.off("change", updateFromModel);
  }, [model]);

  // Sync selection back to Python
  useEffect(() => {
    const currentPySelection = model.get("selected_evals") || [];
    if (JSON.stringify(selectedEvals) !== JSON.stringify(currentPySelection)) {
      model.set("selected_evals", selectedEvals);
      model.save_changes();

      // Trigger callback
      model.set("selection_changed", {
        selected_evals: selectedEvals,
        action: "updated",
        timestamp: Date.now()
      });
      model.save_changes();
    }
  }, [selectedEvals, model]);

  // Filter evaluations based on current filters
  const filteredEvaluations = useMemo(() => {
    if (!evalData?.evaluations) {
      //console.log("🔍 EvalFinder: No eval data or evaluations found", evalData);
      return [];
    }

    //console.log("🔍 EvalFinder: Filtering", evalData.evaluations.length, "evaluations");

    const filtered = evalData.evaluations.filter((evaluation: EvalMetadata) => {
      // Search filter
      if (searchTerm) {
        const searchLower = searchTerm.toLowerCase();
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

    //console.log("🔍 EvalFinder: Filtered to", filtered.length, "evaluations");
    return filtered;
  }, [evalData, searchTerm, categoryFilter]);

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
    setSelectedEvals(prev =>
      prev.includes(evalName)
        ? prev.filter(name => name !== evalName)
        : [...prev, evalName]
    );
  };

  const handleSelectAll = () => {
    const allEvalNames = filteredEvaluations.map((e: EvalMetadata) => e.name);
    setSelectedEvals(allEvalNames);
  };

  const handleClearAll = () => {
    setSelectedEvals([]);
  };

  const handleFilterChange = (filters: {
    category?: string[];
  }) => {
    if (filters.category !== undefined) {
      setCategoryFilter(filters.category);
      model.set("category_filter", filters.category);
    }
    model.save_changes();
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
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
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
            onChange={(e) => setViewMode(e.target.value as "tree" | "list" | "category")}
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
