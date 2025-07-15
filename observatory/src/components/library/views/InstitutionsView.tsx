/**
 * InstitutionsView Component
 * 
 * Displays a grid of institutions/institutions with filtering and sorting capabilities.
 * This component handles the institutions view of the library, including:
 * - Search and filter functionality
 * - Sort controls for various institution metrics
 * - Grid layout of institution cards
 * - Empty state when no results are found
 * 
 * Features:
 * - Real-time filtering by name, location, or tags
 * - Multi-column sorting (name, location, type, members, papers, citations)
 * - Responsive grid layout
 * - Integration with institution cards and overlay modals
 */

import React from 'react';
import { InstitutionCard } from '../cards/InstitutionCard';
import { FilterSortControls } from '../utils';

interface InstitutionsViewProps {
    // Data
    filteredInstitutions: any[];
    
    // State
    searchQuery: string;
    sortBy: 'name' | 'location' | 'type' | 'members' | 'papers' | 'citations';
    sortDirection: 'asc' | 'desc';
    
    // Refs
    filterInputRef: React.RefObject<HTMLInputElement | null>;
    
    // Event handlers
    onSearchChange: (query: string) => void;
    onSortChange: (sortBy: string) => void;
    onSortDirectionChange: (direction: 'asc' | 'desc') => void;
    onCardClick: (institutionId: string) => void;
}

export function InstitutionsView({
    filteredInstitutions,
    searchQuery,
    sortBy,
    sortDirection,
    filterInputRef,
    onSearchChange,
    onSortChange,
    onSortDirectionChange,
    onCardClick
}: InstitutionsViewProps) {
    
    // Sort options configuration for institutions
    const sortOptions = [
        { key: 'name', label: 'Name' },
        { key: 'location', label: 'Location' },
        { key: 'type', label: 'Type' },
        { key: 'members', label: 'Members' },
        { key: 'papers', label: 'Papers' },
        { key: 'citations', label: 'Citations' }
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
                searchPlaceholder="Filter by name, location, or tags..."
                onSearchChange={onSearchChange}
                onSortChange={onSortChange}
                onSortDirectionChange={onSortDirectionChange}
            />
            
            {/* Institutions Grid: left-to-right flow like text */}
            <div className="w-full px-2">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredInstitutions.map(institution => (
                        <InstitutionCard
                            key={institution.id}
                            institution={institution}
                            isAdmin={institution.isAdmin}
                            searchQuery={searchQuery}
                            onCardClick={() => onCardClick(institution.id)}
                        />
                    ))}
                </div>
                
                {/* Empty State */}
                {filteredInstitutions.length === 0 && (
                    <div className="text-center py-8">
                        <svg className="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                        <p className="text-gray-500">No institutions found matching your filter.</p>
                    </div>
                )}
            </div>
        </div>
    );
} 