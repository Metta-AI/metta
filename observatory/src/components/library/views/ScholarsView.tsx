/**
 * ScholarsView Component
 * 
 * Displays a grid of scholars with filtering and sorting capabilities.
 * This component handles the scholars view of the library, including:
 * - Search and filter functionality
 * - Sort controls for various scholar metrics
 * - Grid layout of scholar cards
 * - Empty state when no results are found
 * 
 * Features:
 * - Real-time filtering by name, institution, or expertise
 * - Multi-column sorting (name, affiliation, recent activity, papers, citations, h-index)
 * - Responsive grid layout
 * - Integration with scholar cards and overlay modals
 */

import React from 'react';
import { ScholarCard } from '../cards/ScholarCard';
import { FilterSortControls } from '../utils';

interface ScholarsViewProps {
    // Data
    filteredScholars: any[];
    
    // State
    searchQuery: string;
    sortBy: 'name' | 'affiliation' | 'recentActivity' | 'papers' | 'citations' | 'hIndex';
    sortDirection: 'asc' | 'desc';
    expandedScholarId: string | null;
    
    // Refs
    filterInputRef: React.RefObject<HTMLInputElement | null>;
    
    // Event handlers
    onSearchChange: (query: string) => void;
    onSortChange: (sortBy: string) => void;
    onSortDirectionChange: (direction: 'asc' | 'desc') => void;
    onExpandScholar: (id: string) => void;
    onCollapseScholar: (id: string) => void;
    onToggleFollow: (scholarId: string) => void;
    onTagClick: (tag: string) => void;
    onCardClick: (scholarId: string) => void;
}

export function ScholarsView({
    filteredScholars,
    searchQuery,
    sortBy,
    sortDirection,
    expandedScholarId,
    filterInputRef,
    onSearchChange,
    onSortChange,
    onSortDirectionChange,
    onExpandScholar,
    onCollapseScholar,
    onToggleFollow,
    onTagClick,
    onCardClick
}: ScholarsViewProps) {
    
    // Sort options configuration for scholars
    const sortOptions = [
        { key: 'name', label: 'Name' },
        { key: 'affiliation', label: 'Affiliation' },
        { key: 'recentActivity', label: 'Recent Activity' },
        { key: 'papers', label: 'Papers' },
        { key: 'citations', label: 'Citations' },
        { key: 'hIndex', label: 'H-index' }
    ];



    return (
        <div className="p-4">
            {/* Filter and Sort Controls - using the shared component */}
            <FilterSortControls
                searchQuery={searchQuery}
                sortBy={sortBy}
                sortDirection={sortDirection}
                sortOptions={sortOptions}
                filterInputRef={filterInputRef}
                searchPlaceholder="Filter by name, institution, or expertise..."
                onSearchChange={onSearchChange}
                onSortChange={onSortChange}
                onSortDirectionChange={onSortDirectionChange}
            />
            
            {/* Scholars Grid: left-to-right flow like text */}
            <div className="w-full px-2">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredScholars.map(scholar => (
                        <ScholarCard
                            key={scholar.id}
                            scholar={scholar}
                            expanded={expandedScholarId === scholar.id}
                            onExpand={onExpandScholar}
                            onCollapse={onCollapseScholar}
                            searchQuery={searchQuery}
                            onToggleFollow={onToggleFollow}
                            onTagClick={onTagClick}
                            filterInputRef={filterInputRef}
                            onCardClick={() => onCardClick(scholar.id)}
                        />
                    ))}
                </div>
                
                {/* Empty State */}
                {filteredScholars.length === 0 && (
                    <div className="text-center py-8">
                        <svg className="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                        <p className="text-gray-500">No scholars found matching your filter.</p>
                    </div>
                )}
            </div>
        </div>
    );
} 