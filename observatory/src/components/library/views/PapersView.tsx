/**
 * PapersView Component
 * 
 * Displays a comprehensive table of papers with sorting, filtering, and interactive features.
 * This component handles the papers view of the library, showing papers in a table format
 * with various metadata and user interactions.
 * 
 * The papers table includes:
 * - Paper titles with star/favorite functionality
 * - Author links to scholar profiles (clickable for overlays)
 * - Affiliation links to institution profiles (clickable for overlays)
 * - Research tags for categorization (clickable to filter)
 * - User avatars showing who has read or queued the paper
 * - External links to paper sources
 * - Star ratings and counts
 * 
 * Features:
 * - Sortable columns (all columns are clickable to sort)
 * - Interactive star/favorite toggling
 * - Clickable tags that apply search filters
 * - Clickable authors and affiliations that show overlays
 * - User hover cards showing user details
 * - Responsive table design with horizontal scrolling
 * - Links to related scholars and affiliations
 * 
 * Usage Example:
 * ```tsx
 * <PapersView
 *   papers={papers}
 *   scholars={scholars}
 *   affiliations={affiliations}
 *   onToggleStar={handleToggleStar}
 * />
 * ```
 */

import React, { useState, useMemo } from 'react';
import { UserHoverCard } from '../cards/UserHoverCard';

/**
 * Defines the structure of a paper object
 */
interface Paper {
    id: string;
    title: string;
    starred: boolean;
    authors: string[]; // Array of scholar IDs
    affiliations: string[]; // Array of affiliation IDs
    tags: string[];
    readBy: any[]; // Array of user objects
    queued: any[]; // Array of user objects
    link: string;
    stars: number;
    year?: number; // Optional year field
    citations?: number; // Optional citations field
}

/**
 * Defines the structure of a scholar object
 */
interface Scholar {
    id: string;
    name: string;
}

/**
 * Defines the structure of an affiliation object
 */
interface Affiliation {
    id: string;
    label: string;
}

/**
 * Props that the PapersView component accepts
 */
interface PapersViewProps {
    /** Array of papers to display in the table */
    papers: Paper[];
    
    /** Array of scholars for resolving author IDs to names */
    scholars: Scholar[];
    
    /** Array of affiliations for resolving affiliation IDs to labels */
    affiliations: Affiliation[];
    
    /** Callback function called when user toggles the star/favorite status of a paper */
    onToggleStar: (paperId: string) => void;
    
    /** Current search query for filtering papers */
    searchQuery?: string;
    
    /** Callback function called when search query changes */
    onSearchChange?: (query: string) => void;
    
    /** Reference to the search input for programmatic focus */
    filterInputRef?: React.RefObject<HTMLInputElement | null>;
    
    /** Callback function called when a scholar overlay should be shown */
    onShowScholarOverlay?: (scholarId: string) => void;
    
    /** Callback function called when an affiliation overlay should be shown */
    onShowAffiliationOverlay?: (affiliationId: string) => void;
}

/**
 * PapersView Component
 * 
 * Renders a comprehensive table of papers with sorting, user interactions, and metadata display.
 * The component manages its own sorting state and user hover interactions.
 */
export function PapersView({
    papers,
    scholars,
    affiliations,
    onToggleStar,
    searchQuery = '',
    onSearchChange,
    filterInputRef,
    onShowScholarOverlay,
    onShowAffiliationOverlay
}: PapersViewProps) {
    
    // Validate required props
    if (!Array.isArray(papers)) {
        console.error('PapersView: papers prop must be an array');
        return <div className="p-6 text-red-600">Error: Invalid papers data</div>;
    }
    
    if (!Array.isArray(scholars)) {
        console.error('PapersView: scholars prop must be an array');
        return <div className="p-6 text-red-600">Error: Invalid scholars data</div>;
    }
    
    if (!Array.isArray(affiliations)) {
        console.error('PapersView: affiliations prop must be an array');
        return <div className="p-6 text-red-600">Error: Invalid affiliations data</div>;
    }
    
    if (typeof onToggleStar !== 'function') {
        console.error('PapersView: onToggleStar prop must be a function');
        return <div className="p-6 text-red-600">Error: Invalid onToggleStar callback</div>;
    }
    
    /**
     * State for managing table sorting
     * Controls which column is sorted and in what direction
     */
    const [papersSort, setPapersSort] = useState<{col: string, dir: 'asc'|'desc'}>({col: 'title', dir: 'asc'});
    
    /**
     * State for managing user hover cards
     * Stores the user being hovered and their position for the hover card
     */
    const [hoveredUser, setHoveredUser] = useState<{ user: any, position: { x: number, y: number } } | null>(null);

    /**
     * State for starred filter
     */
    const [showOnlyStarred, setShowOnlyStarred] = useState(false);

    /**
     * State for loading indicator during filtering/sorting operations
     */
    const [isLoading, setIsLoading] = useState(false);

    /**
     * Filter and sort papers based on current filters and search query
     */
    const filteredAndSortedPapers = useMemo(() => {
        // Set loading state for large datasets
        if (papers.length > 100) {
            setIsLoading(true);
            // Use setTimeout to allow UI to update
            setTimeout(() => setIsLoading(false), 0);
        }
        
        // Defensive programming: validate papers array
        if (!Array.isArray(papers)) {
            console.warn('Papers prop is not an array:', papers);
            return [];
        }
        
        let filtered = papers;

        // Filter by search query (title, authors, tags)
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(paper => {
                // Search in title
                if (paper.title.toLowerCase().includes(query)) return true;
                
                // Search in author names
                const authorNames = paper.authors.map(authorId => {
                    const author = scholars.find(s => s.id === authorId);
                    return author ? author.name.toLowerCase() : '';
                });
                if (authorNames.some(name => name.includes(query))) return true;
                
                // Search in tags
                if (paper.tags.some(tag => tag.toLowerCase().includes(query))) return true;
                
                return false;
            });
        }

        // Filter by starred status
        if (showOnlyStarred) {
            filtered = filtered.filter(paper => paper.starred);
        }

        // Sort papers
        return filtered.sort((a, b) => {
            let aValue: string | number;
            let bValue: string | number;
            
            switch (papersSort.col) {
                case 'title':
                    aValue = a.title.toLowerCase();
                    bValue = b.title.toLowerCase();
                    break;
                case 'authors':
                    // Sort by first author name with bounds checking
                    const aFirstAuthor = a.authors.length > 0 ? scholars.find(s => s.id === a.authors[0]) : null;
                    const bFirstAuthor = b.authors.length > 0 ? scholars.find(s => s.id === b.authors[0]) : null;
                    aValue = aFirstAuthor ? aFirstAuthor.name.toLowerCase() : '';
                    bValue = bFirstAuthor ? bFirstAuthor.name.toLowerCase() : '';
                    break;
                case 'affiliations':
                    // Sort by first affiliation label with bounds checking
                    const aFirstAff = a.affiliations.length > 0 ? affiliations.find(aff => aff.id === a.affiliations[0]) : null;
                    const bFirstAff = b.affiliations.length > 0 ? affiliations.find(aff => aff.id === b.affiliations[0]) : null;
                    aValue = aFirstAff ? aFirstAff.label.toLowerCase() : '';
                    bValue = bFirstAff ? bFirstAff.label.toLowerCase() : '';
                    break;
                case 'tags':
                    // Sort by first tag
                    aValue = a.tags.length > 0 ? a.tags[0].toLowerCase() : '';
                    bValue = b.tags.length > 0 ? b.tags[0].toLowerCase() : '';
                    break;
                case 'readBy':
                    // Sort by number of readers
                    aValue = Array.isArray(a.readBy) ? a.readBy.length : 0;
                    bValue = Array.isArray(b.readBy) ? b.readBy.length : 0;
                    break;
                case 'queued':
                    // Sort by number of queued users
                    aValue = Array.isArray(a.queued) ? a.queued.length : 0;
                    bValue = Array.isArray(b.queued) ? b.queued.length : 0;
                    break;
                case 'stars':
                    aValue = a.stars;
                    bValue = b.stars;
                    break;
                default:
                    return 0;
            }
            
            if (aValue < bValue) return papersSort.dir === 'asc' ? -1 : 1;
            if (aValue > bValue) return papersSort.dir === 'asc' ? 1 : -1;
            return 0;
        });
    }, [papers, scholars, affiliations, searchQuery, showOnlyStarred, papersSort]);

    /**
     * Handles sorting when a column header is clicked
     * 
     * When a user clicks a sortable column header:
     * - If it's the same column, we toggle the sort direction
     * - If it's a different column, we switch to that column and set ascending order
     * 
     * @param col - The column name to sort by (e.g., 'title', 'authors', 'stars')
     */
    const handleSort = (col: string) => {
        setPapersSort(prev => {
            // If clicking the same column, toggle direction
            if (prev.col === col) {
                return { col, dir: prev.dir === 'asc' ? 'desc' : 'asc' };
            }
            // If clicking a different column, switch to it with ascending order
            return { col, dir: 'asc' };
        });
    };

    /**
     * Handles mouse enter events on user avatars
     * Shows a hover card with user details at the calculated position
     * 
     * @param e - The mouse event from the avatar element
     * @param user - The user object to display in the hover card
     */
    const handleUserHover = (e: React.MouseEvent<HTMLElement>, user: any) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setHoveredUser({
            user,
            position: {
                x: rect.right + 8, // Position hover card to the right of the avatar
                y: rect.top // Align with the top of the avatar
            }
        });
    };

    /**
     * Handles mouse leave events on user avatars
     * Hides the hover card when mouse leaves the avatar
     */
    const handleUserLeave = () => {
        setHoveredUser(null);
    };

    /**
     * Handles clicking on a tag to apply it as a search filter
     * 
     * When a user clicks on a tag, it writes the tag text into the search box
     * and applies the filter to show papers with that tag
     * 
     * @param tag - The tag text to apply as a search filter
     */
    const handleTagClick = (tag: string) => {
        if (!tag || typeof tag !== 'string') {
            console.warn('Invalid tag provided to handleTagClick:', tag);
            return;
        }
        
        if (onSearchChange) {
            onSearchChange(tag);
        }
    };

    /**
     * Handles clicking on an author to show their overlay
     * 
     * When a user clicks on an author name, it shows the scholar overlay
     * instead of navigating to a URL
     * 
     * @param authorId - The ID of the author to show in the overlay
     */
    const handleAuthorClick = (authorId: string) => {
        if (!authorId || typeof authorId !== 'string') {
            console.warn('Invalid authorId provided to handleAuthorClick:', authorId);
            return;
        }
        
        if (onShowScholarOverlay) {
            onShowScholarOverlay(authorId);
        }
    };

    /**
     * Handles clicking on an affiliation to show its overlay
     * 
     * When a user clicks on an affiliation name, it shows the affiliation overlay
     * instead of navigating to a URL
     * 
     * @param affiliationId - The ID of the affiliation to show in the overlay
     */
    const handleAffiliationClick = (affiliationId: string) => {
        if (!affiliationId || typeof affiliationId !== 'string') {
            console.warn('Invalid affiliationId provided to handleAffiliationClick:', affiliationId);
            return;
        }
        
        if (onShowAffiliationOverlay) {
            onShowAffiliationOverlay(affiliationId);
        }
    };

    /**
     * Renders a user avatar with hover functionality
     * 
     * @param user - The user object to display
     * @param bgColor - Background color class for the avatar
     * @param textColor - Text color class for the avatar
     * @returns JSX element for the user avatar
     */
    const renderUserAvatar = (user: any, bgColor: string, textColor: string) => {
        // Defensive programming: validate user object
        if (!user || typeof user !== 'object') {
            console.warn('Invalid user object provided to renderUserAvatar:', user);
            return null;
        }
        
        const userName = user.name || 'Unknown User';
        const userId = user.id || 'unknown';
        
        return (
            <span
                key={userId}
                className={`inline-flex items-center justify-center w-6 h-6 rounded-full ${bgColor} ${textColor} text-xs font-bold border-2 border-white cursor-pointer`}
                title={userName}
                onMouseEnter={(e) => handleUserHover(e, user)}
                onMouseLeave={handleUserLeave}
            >
                {user.avatar}
            </span>
        );
    };

    /**
     * Renders a sort indicator for table headers
     * 
     * @param columnName - The name of the column to check for sort indicator
     * @returns JSX element for the sort indicator
     */
    const renderSortIndicator = (columnName: string) => (
        <span className="ml-1 align-middle">
            {papersSort.col === columnName ? (papersSort.dir === 'asc' ? '▲' : '▼') : ''}
        </span>
    );

    /**
     * Highlights matching text in a string based on the current search query
     * 
     * This function splits the text by the search query and wraps matching parts
     * in a yellow background highlight span
     * 
     * @param text - The text to highlight
     * @param query - The search query to highlight
     * @returns JSX element with highlighted text
     */
    const highlightText = (text: string, query: string) => {
        if (!query.trim() || !text) {
            return text;
        }
        
        const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedQuery})`, 'gi');
        const parts = text.split(regex);
        
        return parts.map((part, index) => {
            // Check if this part matches the query (case-insensitive)
            if (part.toLowerCase() === query.toLowerCase()) {
                return (
                    <span key={index} className="bg-yellow-200 px-0.5 rounded">
                        {part}
                    </span>
                );
            }
            return part;
        });
    };

    return (
        <div className="p-6">
            <div className="max-w-6xl mx-auto">

                {/* Filter and Sort Controls */}
                <div className="mb-6 space-y-4">
                    {/* Search Input */}
                    <div className="flex items-center gap-4">
                        <div className="flex-1 max-w-md">
                            <input
                                type="text"
                                value={searchQuery}
                                onChange={(e) => {
                                    const value = e.target.value;
                                    // Validate input before calling callback
                                    if (typeof value === 'string' && onSearchChange) {
                                        onSearchChange(value);
                                    }
                                }}
                                placeholder="Filter by title, authors, or tags..."
                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                ref={filterInputRef}
                                maxLength={200} // Prevent extremely long search queries
                            />
                        </div>
                    </div>

                    {/* Starred Filter */}
                    <div className="flex flex-wrap gap-4 items-center">
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={showOnlyStarred}
                                onChange={(e) => setShowOnlyStarred(e.target.checked)}
                                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                            />
                            <span className="text-sm text-gray-700">Show only starred</span>
                        </label>
                    </div>

                </div>
                
                {/* Papers Table */}
                <div className="overflow-x-auto relative">
                    {/* Loading indicator */}
                    {isLoading && (
                        <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center z-20">
                            <div className="text-gray-600">Filtering papers...</div>
                        </div>
                    )}
                    
                    {/* User Hover Card - positioned absolutely based on mouse position */}
                    {hoveredUser && (
                        <UserHoverCard 
                            user={hoveredUser.user} 
                            position={hoveredUser.position} 
                        />
                    )}
                    
                    {/* Main Table */}
                    <table className="min-w-full bg-white border border-gray-200 rounded-lg text-sm">
                        <thead>
                            <tr className="bg-gray-50">
                                {/* Title Column - sortable, sticky left */}
                                <th 
                                    className="px-4 py-2 text-left cursor-pointer hover:bg-gray-100 transition-colors sticky left-0 z-10 bg-white border-r border-gray-200" 
                                    onClick={() => handleSort('title')}
                                >
                                    Title
                                    {renderSortIndicator('title')}
                                </th>
                                
                                {/* Authors Column - sortable */}
                                <th 
                                    className="px-4 py-2 text-left cursor-pointer hover:bg-gray-100 transition-colors" 
                                    onClick={() => handleSort('authors')}
                                >
                                    Authors
                                    {renderSortIndicator('authors')}
                                </th>
                                
                                {/* Affiliations Column - sortable */}
                                <th 
                                    className="px-4 py-2 text-left cursor-pointer hover:bg-gray-100 transition-colors" 
                                    onClick={() => handleSort('affiliations')}
                                >
                                    Affiliations
                                    {renderSortIndicator('affiliations')}
                                </th>
                                
                                {/* Tags Column - sortable */}
                                <th 
                                    className="px-4 py-2 text-left cursor-pointer hover:bg-gray-100 transition-colors" 
                                    onClick={() => handleSort('tags')}
                                >
                                    Tags
                                    {renderSortIndicator('tags')}
                                </th>
                                
                                {/* Read by Column - sortable */}
                                <th 
                                    className="px-4 py-2 text-left cursor-pointer hover:bg-gray-100 transition-colors" 
                                    onClick={() => handleSort('readBy')}
                                >
                                    Read by
                                    {renderSortIndicator('readBy')}
                                </th>
                                
                                {/* Link Column - not sortable */}
                                <th className="px-4 py-2 text-left">Link</th>
                                
                                {/* Queued Column - sortable */}
                                <th 
                                    className="px-4 py-2 text-left cursor-pointer hover:bg-gray-100 transition-colors" 
                                    onClick={() => handleSort('queued')}
                                >
                                    Queued
                                    {renderSortIndicator('queued')}
                                </th>
                                
                                {/* Stars Column - sortable */}
                                <th 
                                    className="px-4 py-2 text-left cursor-pointer hover:bg-gray-100 transition-colors" 
                                    onClick={() => handleSort('stars')}
                                >
                                    Stars
                                    {renderSortIndicator('stars')}
                                </th>
                            </tr>
                        </thead>
                        
                        <tbody>
                            {filteredAndSortedPapers.map(paper => {
                                // Validate paper object before rendering
                                if (!paper || typeof paper !== 'object' || !paper.id) {
                                    console.warn('Invalid paper object found:', paper);
                                    return null;
                                }
                                
                                return (
                                <tr key={paper.id} className="border-t border-gray-100 hover:bg-gray-50">
                                    {/* Title Cell - sticky left with star button */}
                                    <td className="px-4 py-2 whitespace-nowrap flex items-center gap-2 sticky left-0 z-10 bg-white border-r border-gray-200">
                                        <button 
                                            onClick={() => onToggleStar(paper.id)} 
                                            className="focus:outline-none hover:scale-110 transition-transform"
                                            aria-label={paper.starred ? 'Remove from favorites' : 'Add to favorites'}
                                        >
                                            {paper.starred ? (
                                                <svg className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                                                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z"/>
                                                </svg>
                                            ) : (
                                                <svg className="w-4 h-4 text-gray-300 hover:text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 20 20">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 15l-5.878 3.09 1.122-6.545L.488 6.91l6.564-.955L10 0l2.948 5.955 6.564.955-4.756 4.635 1.122 6.545z"/>
                                                </svg>
                                            )}
                                        </button>
                                        <span>{highlightText(paper.title, searchQuery)}</span>
                                    </td>
                                    
                                    {/* Authors Cell - clickable links to scholar overlays */}
                                    <td className="px-4 py-2 whitespace-nowrap">
                                        {paper.authors.map((authorId: string, idx: number) => {
                                            const author = scholars.find(s => s.id === authorId);
                                            if (!author) {
                                                console.warn(`Scholar with ID "${authorId}" not found for paper "${paper.title}"`);
                                                return (
                                                    <span key={`unknown-${authorId}-${idx}`} className="text-gray-400 italic mr-1">
                                                        Unknown Author{idx < paper.authors.length - 1 ? ',' : ''}
                                                    </span>
                                                );
                                            }
                                            return (
                                                <button
                                                    key={author.id}
                                                    onClick={() => handleAuthorClick(author.id)}
                                                    className="text-primary-600 hover:text-primary-700 hover:underline mr-1 cursor-pointer bg-transparent border-none p-0 font-inherit"
                                                >
                                                    {highlightText(author.name, searchQuery)}{idx < paper.authors.length - 1 ? ',' : ''}
                                                </button>
                                            );
                                        })}
                                    </td>
                                    
                                    {/* Affiliations Cell - clickable links to affiliation overlays */}
                                    <td className="px-4 py-2 whitespace-nowrap">
                                        {paper.affiliations.map((affId: string, idx: number) => {
                                            const aff = affiliations.find(a => a.id === affId);
                                            if (!aff) {
                                                console.warn(`Affiliation with ID "${affId}" not found for paper "${paper.title}"`);
                                                return (
                                                    <span key={`unknown-${affId}-${idx}`} className="text-gray-400 italic mr-1">
                                                        Unknown Institution{idx < paper.affiliations.length - 1 ? ',' : ''}
                                                    </span>
                                                );
                                            }
                                            return (
                                                <button
                                                    key={aff.id}
                                                    onClick={() => handleAffiliationClick(aff.id)}
                                                    className="text-primary-600 hover:text-primary-700 hover:underline mr-1 cursor-pointer bg-transparent border-none p-0 font-inherit"
                                                >
                                                    {highlightText(aff.label, searchQuery)}{idx < paper.affiliations.length - 1 ? ',' : ''}
                                                </button>
                                            );
                                        })}
                                    </td>
                                    
                                    {/* Tags Cell - clickable research area tags */}
                                    <td className="px-4 py-2 whitespace-nowrap">
                                        {paper.tags.map((tag, idx) => (
                                            <button
                                                key={idx}
                                                onClick={() => handleTagClick(tag)}
                                                className="inline-block bg-gray-100 text-gray-700 text-xs rounded-full px-2 py-0.5 mr-1 mb-0.5 hover:bg-gray-200 hover:text-gray-800 transition-colors cursor-pointer"
                                                title={`Click to filter by "${tag}"`}
                                            >
                                                {highlightText(tag, searchQuery)}
                                            </button>
                                        ))}
                                    </td>
                                    
                                    {/* Read By Cell - user avatars with hover cards */}
                                    <td className="px-4 py-2 whitespace-nowrap">
                                        <div className="flex -space-x-2">
                                            {Array.isArray(paper.readBy) && paper.readBy.length > 0 ? 
                                                paper.readBy.map((user: any) => 
                                                    renderUserAvatar(user, 'bg-primary-200', 'text-primary-800')
                                                ) : null
                                            }
                                        </div>
                                    </td>
                                    
                                    {/* Link Cell - external paper link */}
                                    <td className="px-4 py-2 whitespace-nowrap">
                                        <a 
                                            href={paper.link} 
                                            target="_blank" 
                                            rel="noopener noreferrer" 
                                            className="text-primary-500 hover:text-primary-600"
                                            aria-label="Open paper in new tab"
                                        >
                                            <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 3h7v7m0 0L10 21l-7-7 11-11z"/>
                                            </svg>
                                        </a>
                                    </td>
                                    
                                    {/* Queued Cell - user avatars for queued papers */}
                                    <td className="px-4 py-2 whitespace-nowrap">
                                        <div className="flex -space-x-2">
                                            {Array.isArray(paper.queued) && paper.queued.length > 0 ? 
                                                paper.queued.map((user: any) => 
                                                    renderUserAvatar(user, 'bg-primary-100', 'text-primary-700')
                                                ) : null
                                            }
                                        </div>
                                    </td>
                                    
                                    {/* Stars Cell - star count */}
                                    <td className="px-4 py-2 whitespace-nowrap text-center">{paper.stars}</td>
                                </tr>
                            );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
} 