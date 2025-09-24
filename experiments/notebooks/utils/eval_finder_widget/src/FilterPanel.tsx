import React from "react";

interface FilterPanelProps {
  categoryFilter: string[];
  availableCategories: string[];
  onFilterChange: (filters: {
    category?: string[];
  }) => void;
}

export const FilterPanel: React.FC<FilterPanelProps> = ({
  categoryFilter,
  availableCategories,
  onFilterChange
}) => {

  const handleCategoryToggle = (category: string) => {
    const newFilter = categoryFilter.includes(category)
      ? categoryFilter.filter(c => c !== category)
      : [...categoryFilter, category];
    onFilterChange({ category: newFilter });
  };

  return (
    <div className="filter-panel">
      <div className="filter-section">
        <label className="filter-label">Categories:</label>
        <div className="checkbox-group">
          {availableCategories.map((category) => (
            <label key={category} className="checkbox-item">
              <input
                type="checkbox"
                checked={categoryFilter.length === 0 || categoryFilter.includes(category)}
                onChange={() => handleCategoryToggle(category)}
              />
              <span className="category-label">
                {category.replace('_', ' ')}
              </span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
};
