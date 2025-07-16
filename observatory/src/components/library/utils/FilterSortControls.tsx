/**
 * FilterSortControls Component
 * 
 * A reusable component that provides search filtering and sorting controls for data grids.
 * This component is used across multiple views (scholars, institutions, etc.) to provide
 * consistent filtering and sorting functionality.
 * 
 * The component consists of two main parts:
 * 1. A search input field with a search icon
 * 2. A set of sort buttons that can be toggled to change sort order
 * 
 * Features:
 * - Real-time search input with placeholder text
 * - Configurable sort options (passed as props)
 * - Visual feedback for active sort option and direction
 * - Responsive design that works on different screen sizes
 * - Accessible keyboard navigation and screen reader support
 * 
 * Usage Example:
 * ```tsx
 * <FilterSortControls
 *   searchQuery="john"
 *   sortBy="name"
 *   sortDirection="asc"
 *   sortOptions={[
 *     { key: 'name', label: 'Name' },
 *     { key: 'date', label: 'Date' }
 *   ]}
 *   onSearchChange={setSearchQuery}
 *   onSortChange={setSortBy}
 *   onSortDirectionChange={setSortDirection}
 *   searchPlaceholder="Filter by name or description..."
 * />
 * ```
 */

import React from 'react';

/**
 * Defines the structure of a single sort option
 */
interface SortOption {
    /** Unique identifier for the sort option (e.g., 'name', 'date', 'score') */
    key: string;
    /** Human-readable label to display on the sort button (e.g., 'Name', 'Date', 'Score') */
    label: string;
}

/**
 * Props that the FilterSortControls component accepts
 */
interface FilterSortControlsProps {
    /** Current search query text - this controls what's displayed in the search input */
    searchQuery: string;
    
    /** Currently active sort option - should match one of the keys in sortOptions */
    sortBy: string;
    
    /** Current sort direction - either 'asc' (ascending) or 'desc' (descending) */
    sortDirection: 'asc' | 'desc';
    
    /** Array of available sort options - each option has a key and display label */
    sortOptions: SortOption[];
    
    /** Reference to the search input element - useful for programmatic focus or other DOM operations */
    filterInputRef: React.RefObject<HTMLInputElement | null>;
    
    /** Placeholder text to show in the search input when it's empty */
    searchPlaceholder: string;
    
    /** Callback function called whenever the user types in the search input */
    onSearchChange: (query: string) => void;
    
    /** Callback function called when user clicks a sort button to change the sort field */
    onSortChange: (sortBy: string) => void;
    
    /** Callback function called when user clicks a sort button to change the sort direction */
    onSortDirectionChange: (direction: 'asc' | 'desc') => void;
}

/**
 * FilterSortControls Component
 * 
 * Renders a search input and sort controls that can be used to filter and sort data.
 * The component is designed to be flexible and reusable across different data types.
 */
export function FilterSortControls({
    searchQuery,
    sortBy,
    sortDirection,
    sortOptions,
    filterInputRef,
    searchPlaceholder,
    onSearchChange,
    onSortChange,
    onSortDirectionChange
}: FilterSortControlsProps) {
    
    /**
     * Handles clicks on sort buttons
     * 
     * When a user clicks a sort button:
     * - If it's the same sort option that's already active, we toggle the direction
     * - If it's a different sort option, we switch to that option and set direction to ascending
     * 
     * @param key - The sort option key that was clicked
     */
    const handleSortClick = (key: string) => {
        if (sortBy === key) {
            // Same sort option clicked - toggle direction
            onSortDirectionChange(sortDirection === 'asc' ? 'desc' : 'asc');
        } else {
            // Different sort option clicked - switch to it and set ascending
            onSortChange(key);
            onSortDirectionChange('asc');
        }
    };

    return (
        <div className="w-full mb-4 space-y-4">
            {/* Search Input Section */}
            <div className="relative">
                {/* Search Icon - positioned absolutely in the left side of the input */}
                <svg 
                    className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                    aria-hidden="true" // Hide from screen readers since it's decorative
                >
                    <circle cx="11" cy="11" r="8" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m21 21-4.35-4.35" />
                </svg>
                
                {/* Search Input Field */}
                <input
                    ref={filterInputRef}
                    type="text"
                    placeholder="Filter..."
                    value={searchQuery}
                    onChange={(e) => onSearchChange(e.target.value)}
                    className="w-full pl-10 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    aria-label="Search and filter items" // Accessibility label for screen readers
                />
                
                {/* Clear Button - only show when there's content */}
                {searchQuery && (
                    <button
                        onClick={() => onSearchChange('')}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 hover:text-gray-600 transition-colors"
                        aria-label="Clear search"
                        type="button"
                    >
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                )}
            </div>
            
            {/* Sort Controls Section */}
            <div className="flex items-center gap-4 text-sm">
                {/* Sort Label */}
                <span className="text-gray-600 font-medium">Sort by:</span>
                
                {/* Sort Buttons Container */}
                <div className="flex gap-2">
                    {sortOptions.map(({ key, label }) => {
                        // Determine if this sort option is currently active
                        const isActive = sortBy === key;
                        
                        return (
                            <button
                                key={key}
                                onClick={() => handleSortClick(key)}
                                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                                    isActive
                                        ? 'bg-primary-100 text-primary-700 border border-primary-200'
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-gray-200'
                                }`}
                                aria-pressed={isActive} // Accessibility: indicates if button is pressed/active
                                aria-label={`Sort by ${label} ${isActive ? `(${sortDirection === 'asc' ? 'ascending' : 'descending'})` : ''}`}
                            >
                                {label}
                                {/* Show sort direction indicator only for the active sort option */}
                                {isActive && (
                                    <span className="ml-1" aria-label={`sorted ${sortDirection === 'asc' ? 'ascending' : 'descending'}`}>
                                        {sortDirection === 'asc' ? '↑' : '↓'}
                                    </span>
                                )}
                            </button>
                        );
                    })}
                </div>
            </div>
        </div>
    );
} 