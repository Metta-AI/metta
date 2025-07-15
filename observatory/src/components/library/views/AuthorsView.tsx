/**
 * AuthorsView Component
 * 
 * Displays a grid of authors with filtering and sorting capabilities.
 * This component handles the authors view of the library, including:
 * - Search and filter functionality
 * - Sort controls for various author metrics
 * - Grid layout of author cards
 * - Empty state when no results are found
 * 
 * Features:
 * - Real-time filtering by name, institution, or expertise
 * - Multi-column sorting (name, institution, recent activity, papers, citations, h-index)
 * - Responsive grid layout
 * - Integration with author cards and overlay modals
 */

import React from 'react';
import { AuthorCard } from '../cards/AuthorCard';
import { FilterSortControls } from '../utils';

interface AuthorsViewProps {
    // Data
    filteredAuthors: any[];
    
    // State
    searchQuery: string;
    sortBy: 'name' | 'institution' | 'recentActivity' | 'papers' | 'citations' | 'hIndex';
    sortDirection: 'asc' | 'desc';
    expandedAuthorId: string | null;
    
    // Refs
    filterInputRef: React.RefObject<HTMLInputElement | null>;
    
    // Event handlers
    onSearchChange: (query: string) => void;
    onSortChange: (sortBy: string) => void;
    onSortDirectionChange: (direction: 'asc' | 'desc') => void;
    onExpandAuthor: (id: string) => void;
    onCollapseAuthor: (id: string) => void;
    onToggleFollow: (authorId: string) => void;
    onTagClick: (tag: string) => void;
    onCardClick: (authorId: string) => void;
}

export function AuthorsView({
    filteredAuthors,
    searchQuery,
    sortBy,
    sortDirection,
    expandedAuthorId,
    filterInputRef,
    onSearchChange,
    onSortChange,
    onSortDirectionChange,
    onExpandAuthor,
    onCollapseAuthor,
    onToggleFollow,
    onTagClick,
    onCardClick
}: AuthorsViewProps) {
    
    // Sort options configuration for authors
    const sortOptions = [
        { key: 'name', label: 'Name' },
        { key: 'institution', label: 'Institution' },
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
            
            {/* Authors Grid: left-to-right flow like text */}
            <div className="w-full px-2">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredAuthors.map(author => (
                        <AuthorCard
                            key={author.id}
                            author={author}
                            expanded={expandedAuthorId === author.id}
                            onExpand={onExpandAuthor}
                            onCollapse={onCollapseAuthor}
                            searchQuery={searchQuery}
                            onToggleFollow={onToggleFollow}
                            onTagClick={onTagClick}
                            filterInputRef={filterInputRef}
                            onCardClick={() => onCardClick(author.id)}
                        />
                    ))}
                </div>
                
                {/* Empty State */}
                {filteredAuthors.length === 0 && (
                    <div className="text-center py-8">
                        <svg className="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                        <p className="text-gray-500">No authors found matching your filter.</p>
                    </div>
                )}
            </div>
        </div>
    );
} 