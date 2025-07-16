/**
 * PapersView Component
 * 
 * Displays a comprehensive table of papers with sorting, filtering, and interactive features.
 * This component uses native table semantics with sticky positioning for frozen columns,
 * providing robust cross-browser compatibility and proper accessibility.
 * 
 * Implementation Details:
 * - Native <table> wrapped in scroll container with overflow: auto
 * - <colgroup> with <col> elements for column width management
 * - position: sticky for frozen header row and first column
 * - Absolutely positioned resize handles on each column
 * - Semantic table structure maintained for accessibility
 * 
 * Features:
 * - Sortable columns (all columns are clickable to sort)
 * - Interactive star/favorite toggling
 * - Clickable tags that apply search filters
 * - Clickable authors and institutions that show overlays
 * - User hover cards showing user details
 * - Frozen title column with horizontal scrolling for other columns
 * - All columns individually resizable with proper handles
 * - Links to related scholars and institutions
 * - Hover tooltips for long titles
 * 
 * Usage Example:
 * ```tsx
 * <PapersView
 *   papers={papers}
 *   scholars={scholars}
 *   institutions={institutions}
 *   onToggleStar={handleToggleStar}
 * />
 * ```
 */

import React, { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import { UserHoverCard } from '../cards/UserHoverCard';
import { PaperOverlay } from '../cards/PaperOverlay';

/**
 * Utility function to validate if a string is a valid URL
 * @param url - The URL string to validate
 * @returns boolean indicating if the URL is valid
 */
const isValidUrl = (url: string): boolean => {
    if (!url || typeof url !== 'string') {
        return false;
    }
    
    try {
        const urlObj = new URL(url);
        return urlObj.protocol === 'http:' || urlObj.protocol === 'https:';
    } catch {
        return false;
    }
};

/**
 * Defines the structure of a paper object
 */
interface Paper {
    id: string;
    title: string;
    starred: boolean;
    authors: string[]; // Array of scholar IDs
    institutions: string[]; // Array of institution IDs
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
 * Defines the structure of an institution object
 */
interface Institution {
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
    
    /** Array of institutions for resolving institution IDs to labels */
    institutions: Institution[];
    
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
    
    /** Callback function called when an institution overlay should be shown */
    onShowInstitutionOverlay?: (institutionId: string) => void;
}

/**
 * Column configuration for the table
 */
interface ColumnConfig {
    key: string;
    label: string;
    width: number;
    minWidth: number;
    maxWidth: number;
    sortable: boolean;
    sticky?: boolean; // For frozen column
    renderHeader: (sortIndicator: React.ReactNode) => React.ReactNode;
    renderCell: (paper: Paper) => React.ReactNode;
}

/**
 * PapersView Component
 * 
 * Renders a comprehensive table of papers using native table semantics
 * with sticky positioning for frozen columns and proper accessibility.
 */
export function PapersView({
    papers,
    scholars,
    institutions,
    onToggleStar,
    searchQuery = '',
    onSearchChange,
    filterInputRef,
    onShowScholarOverlay,
    onShowInstitutionOverlay
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
    
    if (!Array.isArray(institutions)) {
        console.error('PapersView: institutions prop must be an array');
        return <div className="p-6 text-red-600">Error: Invalid institutions data</div>;
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
     * State for column widths (all resizable)
     * Title column: 400px (frozen)
     * Other columns: reasonable defaults
     */
    const [columnWidths, setColumnWidths] = useState({
        title: 400,
        tags: 300,
        readBy: 120,
        link: 80,
        queued: 120,
        stars: 80
    });

    /**
     * Drag state for column resizing
     */
    const [isDragging, setIsDragging] = useState(false);
    const isDraggingRef = useRef(false);
    const dragStartX = useRef(0);
    const dragStartWidth = useRef(0);
    const [mouseX, setMouseX] = useState(0);
    const draggedColumnRef = useRef<string | null>(null);
    const tableRef = useRef<HTMLTableElement>(null);

    /**
     * State for hover tooltip
     */
    const [hoveredTitle, setHoveredTitle] = useState<{ text: string, position: { x: number, y: number } } | null>(null);

    /**
     * State for paper overlay
     */
    const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);

    /**
     * Handle mouse down on any column resize handle
     */
    const handleMouseDown = useCallback((e: React.MouseEvent, columnName: string) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
        isDraggingRef.current = true;
        draggedColumnRef.current = columnName;
        dragStartX.current = e.clientX;
        dragStartWidth.current = columnWidths[columnName as keyof typeof columnWidths];
        
        // Add global mouse event listeners
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    }, [columnWidths]);

    /**
     * Handle mouse move during column resize
     */
    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (!isDraggingRef.current || !draggedColumnRef.current) {
            return;
        }
        
        const deltaX = e.clientX - dragStartX.current;
        const newWidth = Math.max(100, Math.min(800, dragStartWidth.current + deltaX));
        
        setColumnWidths(prev => ({
            ...prev,
            [draggedColumnRef.current!]: newWidth
        }));
        setMouseX(e.clientX);
    }, []);

    /**
     * Handle mouse up to end column resize
     */
    const handleMouseUp = useCallback(() => {
        setIsDragging(false);
        isDraggingRef.current = false;
        draggedColumnRef.current = null;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    }, [handleMouseMove]);

    /**
     * Cleanup event listeners on unmount
     */
    useEffect(() => {
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, []); // Empty dependency array since we only need cleanup on unmount

    /**
     * Handle title hover for tooltip
     */
    const handleTitleHover = useCallback((e: React.MouseEvent, title: string) => {
        const element = e.currentTarget as HTMLElement;
        const rect = element.getBoundingClientRect();
        
        // Only show tooltip if text is truncated
        if (element.scrollWidth > element.clientWidth) {
            setHoveredTitle({
                text: title,
                position: { x: rect.left + rect.width / 2, y: rect.bottom + 5 }
            });
        }
    }, []);

    /**
     * Handle title leave to hide tooltip
     */
    const handleTitleLeave = useCallback(() => {
        setHoveredTitle(null);
    }, []);

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
                case 'institutions':
                    // Sort by first institution label with bounds checking
                    const aFirstAff = a.institutions.length > 0 ? institutions.find(aff => aff.id === a.institutions[0]) : null;
                    const bFirstAff = b.institutions.length > 0 ? institutions.find(aff => aff.id === b.institutions[0]) : null;
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
                    // Sort by star count
                    aValue = a.stars || 0;
                    bValue = b.stars || 0;
                    break;
                default:
                    aValue = a.title.toLowerCase();
                    bValue = b.title.toLowerCase();
            }
            
            // Handle string comparison
            if (typeof aValue === 'string' && typeof bValue === 'string') {
                return papersSort.dir === 'asc' 
                    ? aValue.localeCompare(bValue)
                    : bValue.localeCompare(aValue);
            }
            
            // Handle number comparison
            if (typeof aValue === 'number' && typeof bValue === 'number') {
                return papersSort.dir === 'asc' ? aValue - bValue : bValue - aValue;
            }
            
            return 0;
        });
    }, [papers, searchQuery, showOnlyStarred, papersSort, scholars, institutions]);

    /**
     * Handle column sorting
     */
    const handleSort = (col: string) => {
        setPapersSort(prev => ({
            col,
            dir: prev.col === col && prev.dir === 'asc' ? 'desc' : 'asc'
        }));
    };

    /**
     * Handle user hover for hover cards
     */
    const handleUserHover = (e: React.MouseEvent<HTMLElement>, user: any) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setHoveredUser({
            user,
            position: { x: rect.left, y: rect.bottom + 5 }
        });
    };

    /**
     * Handle user leave to hide hover cards
     */
    const handleUserLeave = () => {
        setHoveredUser(null);
    };

    /**
     * Handle tag click to apply search filter
     */
    const handleTagClick = (tag: string) => {
        if (!tag || typeof tag !== 'string') {
            console.warn('Invalid tag provided to handleTagClick:', tag);
            return;
        }
        
        const trimmedTag = tag.trim();
        if (!trimmedTag) {
            console.warn('Empty tag provided to handleTagClick');
            return;
        }
        
        if (onSearchChange) {
            onSearchChange(trimmedTag);
        }
    };

    /**
     * Handle author click to show scholar overlay
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
     * Handle institution click to show institution overlay
     */
    const handleInstitutionClick = (institutionId: string) => {
        if (!institutionId || typeof institutionId !== 'string') {
            console.warn('Invalid institutionId provided to handleInstitutionClick:', institutionId);
            return;
        }
        
        if (onShowInstitutionOverlay) {
            onShowInstitutionOverlay(institutionId);
        }
    };

    /**
     * Handle paper title click to show paper overlay
     */
    const handlePaperClick = (paper: Paper) => {
        if (!paper || typeof paper !== 'object' || !paper.id) {
            console.warn('Invalid paper object provided to handlePaperClick:', paper);
            return;
        }
        
        setSelectedPaper(paper);
    };

    /**
     * Handle paper overlay close
     */
    const handlePaperOverlayClose = () => {
        setSelectedPaper(null);
    };

    /**
     * Handle toggle queue status
     */
    const handleToggleQueue = (paperId: string) => {
        if (!paperId || typeof paperId !== 'string') {
            console.warn('Invalid paperId provided to handleToggleQueue:', paperId);
            return;
        }
        
        // TODO: Implement queue toggle functionality
        console.log('Toggle queue for paper:', paperId);
    };

    /**
     * Render user avatar with hover functionality
     */
    const renderUserAvatar = (user: any, bgColor: string, textColor: string) => {
        // Defensive programming: validate user object
        if (!user || typeof user !== 'object') {
            console.warn('Invalid user object provided to renderUserAvatar:', user);
            return null;
        }
        
        const userName = user.name || 'Unknown User';
        const userId = user.id || 'unknown';
        
        // Safely generate initials
        const initials = typeof userName === 'string' 
            ? userName.split(' ').map((n: string) => n[0]).join('').toUpperCase().slice(0, 2)
            : '??';
        
        return (
            <div
                key={userId}
                className={`w-6 h-6 rounded-full ${bgColor} ${textColor} text-xs font-medium flex items-center justify-center cursor-pointer border border-white hover:scale-110 transition-transform`}
                onMouseEnter={(e) => handleUserHover(e, user)}
                onMouseLeave={handleUserLeave}
                title={userName}
            >
                {user.avatar || initials}
            </div>
        );
    };

    /**
     * Render sort indicator for column headers
     */
    const renderSortIndicator = (columnName: string) => (
        <span className="ml-1">
            {papersSort.col === columnName ? (
                papersSort.dir === 'asc' ? '↑' : '↓'
            ) : (
                <span className="text-gray-400">↕</span>
            )}
        </span>
    );

    /**
     * Highlight search query in text
     */
    const highlightText = (text: string, query: string) => {
        if (!text || typeof text !== 'string') {
            return text;
        }
        
        if (!query || typeof query !== 'string' || !query.trim()) {
            return text;
        }
        
        try {
            const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const parts = text.split(new RegExp(`(${escapedQuery})`, 'gi'));
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
        } catch (error) {
            console.warn('Error highlighting text:', error);
            return text;
        }
    };

    /**
     * Column configurations for the table
     */
    const columnConfigs: ColumnConfig[] = useMemo(() => [
        {
            key: 'title',
            label: 'Title',
            width: columnWidths.title,
            minWidth: 200,
            maxWidth: 800,
            sortable: true,
            sticky: true, // Frozen column
            renderHeader: (sortIndicator) => (
                <div className="flex items-center justify-between">
                    <span>Title</span>
                    {sortIndicator}
                </div>
            ),
            renderCell: (paper) => (
                <div className="flex items-center gap-2">
                    <button 
                        onClick={() => onToggleStar(paper.id)} 
                        className="focus:outline-none hover:scale-110 transition-transform flex-shrink-0"
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
                    <button
                        className="truncate block text-left hover:text-primary-600 transition-colors"
                        onMouseEnter={(e) => handleTitleHover(e, paper.title)}
                        onMouseLeave={handleTitleLeave}
                        onClick={() => handlePaperClick(paper)}
                        title={paper.title}
                    >
                        {highlightText(paper.title, searchQuery)}
                    </button>
                </div>
            )
        },
        {
            key: 'tags',
            label: 'Tags',
            width: columnWidths.tags,
            minWidth: 100,
            maxWidth: 300,
            sortable: true,
            renderHeader: (sortIndicator) => (
                <div className="flex items-center justify-between">
                    <span>Tags</span>
                    {sortIndicator}
                </div>
            ),
            renderCell: (paper) => (
                <div className="whitespace-nowrap">
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
                </div>
            )
        },
        {
            key: 'readBy',
            label: 'Read by',
            width: columnWidths.readBy,
            minWidth: 100,
            maxWidth: 200,
            sortable: true,
            renderHeader: (sortIndicator) => (
                <div className="flex items-center justify-between">
                    <span>Read by</span>
                    {sortIndicator}
                </div>
            ),
            renderCell: (paper) => (
                <div className="flex -space-x-2">
                    {Array.isArray(paper.readBy) && paper.readBy.length > 0 ? 
                        paper.readBy.map((user: any) => 
                            renderUserAvatar(user, 'bg-primary-200', 'text-primary-800')
                        ) : null
                    }
                </div>
            )
        },
        {
            key: 'link',
            label: 'Link',
            width: columnWidths.link,
            minWidth: 60,
            maxWidth: 100,
            sortable: false,
            renderHeader: () => <span>Link</span>,
            renderCell: (paper) => (
                paper.link && isValidUrl(paper.link) ? (
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
                ) : null
            )
        },
        {
            key: 'queued',
            label: 'Queued',
            width: columnWidths.queued,
            minWidth: 100,
            maxWidth: 200,
            sortable: true,
            renderHeader: (sortIndicator) => (
                <div className="flex items-center justify-between">
                    <span>Queued</span>
                    {sortIndicator}
                </div>
            ),
            renderCell: (paper) => (
                <div className="flex -space-x-2">
                    {Array.isArray(paper.queued) && paper.queued.length > 0 ? 
                        paper.queued.map((user: any) => 
                            renderUserAvatar(user, 'bg-primary-100', 'text-primary-700')
                        ) : null
                    }
                </div>
            )
        },
        {
            key: 'stars',
            label: 'Stars',
            width: columnWidths.stars,
            minWidth: 60,
            maxWidth: 100,
            sortable: true,
            renderHeader: (sortIndicator) => (
                <div className="flex items-center justify-between">
                    <span>Stars</span>
                    {sortIndicator}
                </div>
            ),
            renderCell: (paper) => (
                <div className="text-center">
                    {paper.stars > 0 ? (
                        <div className="relative inline-flex items-center justify-center">
                            <svg className="w-8 h-8 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z"/>
                            </svg>
                            <span className="absolute text-sm font-medium text-black">
                                {paper.stars}
                            </span>
                        </div>
                    ) : null}
                </div>
            )
        }
    ], [columnWidths, papersSort, searchQuery, scholars, institutions, onToggleStar, handleTitleHover, handleTitleLeave, handleAuthorClick, handleInstitutionClick, handleTagClick, renderUserAvatar, highlightText]);

    return (
        <div className="p-4">
            <div className="w-full px-2">
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
                
                {/* Papers Table Container */}
                <div className="relative">
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

                    {/* Title Tooltip - positioned absolutely based on mouse position */}
                    {hoveredTitle && (
                        <div 
                            className="absolute z-30 bg-gray-900 text-white text-sm px-3 py-2 rounded-lg shadow-lg max-w-md break-words"
                            style={{
                                left: hoveredTitle.position.x,
                                top: hoveredTitle.position.y,
                                transform: 'translateX(-50%)'
                            }}
                        >
                            {hoveredTitle.text}
                        </div>
                    )}

                    {/* Width indicator during drag */}
                    {isDragging && draggedColumnRef.current && (
                        <div 
                            className="fixed z-50 bg-blue-500 text-white text-xs px-2 py-1 rounded pointer-events-none"
                            style={{
                                left: mouseX,
                                top: 10
                            }}
                        >
                            {columnWidths[draggedColumnRef.current as keyof typeof columnWidths]}px
                        </div>
                    )}

                    {/* Paper Overlay */}
                    {selectedPaper && (
                        <PaperOverlay
                            paper={selectedPaper}
                            scholars={scholars}
                            institutions={institutions}
                            onToggleStar={onToggleStar}
                            onToggleQueue={handleToggleQueue}
                            onShowAuthorOverlay={onShowScholarOverlay || (() => {})}
                            onShowInstitutionOverlay={onShowInstitutionOverlay || (() => {})}
                            onClose={handlePaperOverlayClose}
                        />
                    )}
                    
                    {/* Scroll Container with Table */}
                    <div className="overflow-auto border border-gray-200 rounded-lg">
                        <table 
                            ref={tableRef}
                            className="w-full bg-white text-sm"
                            style={{ tableLayout: 'fixed' }}
                        >
                            {/* Column Group for Width Management */}
                            <colgroup>
                                {columnConfigs.map((config) => (
                                    <col 
                                        key={config.key}
                                        style={{ width: `${config.width}px` }}
                                    />
                                ))}
                            </colgroup>
                            
                            {/* Table Header */}
                            <thead>
                                <tr className="bg-gray-50">
                                    {columnConfigs.map((config) => (
                                        <th 
                                            key={config.key}
                                            className={`px-4 py-2 text-left relative ${
                                                config.sticky 
                                                    ? 'sticky left-0 z-10 bg-gray-50' 
                                                    : ''
                                            } ${
                                                config.sortable 
                                                    ? 'cursor-pointer hover:bg-gray-100 transition-colors' 
                                                    : ''
                                            }`}
                                            style={{
                                                position: config.sticky ? 'sticky' : 'static',
                                                left: config.sticky ? 0 : 'auto',
                                                top: 0,
                                                zIndex: config.sticky ? 10 : 'auto'
                                            }}
                                            onClick={() => {
                                                if (config.sortable && !isDragging) {
                                                    handleSort(config.key);
                                                }
                                            }}
                                        >
                                            <div className="flex items-center justify-between">
                                                {config.renderHeader(renderSortIndicator(config.key))}
                                            </div>
                                            {/* Resize handle */}
                                            <div
                                                className="absolute right-0 top-0 bottom-0 w-2 cursor-col-resize hover:bg-blue-400 transition-colors z-20"
                                                onMouseDown={(e) => handleMouseDown(e, config.key)}
                                                title="Drag to resize column"
                                                style={{ 
                                                    cursor: 'col-resize',
                                                    userSelect: 'none'
                                                }}
                                            />
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            
                            {/* Table Body */}
                            <tbody>
                                {filteredAndSortedPapers.map(paper => {
                                    // Validate paper object before rendering
                                    if (!paper || typeof paper !== 'object' || !paper.id) {
                                        console.warn('Invalid paper object found:', paper);
                                        return null;
                                    }
                                    
                                    return (
                                        <tr key={paper.id} className="border-t border-gray-100 hover:bg-gray-50">
                                            {columnConfigs.map((config) => (
                                                <td 
                                                    key={config.key}
                                                    className={`px-4 py-2 ${
                                                        config.sticky 
                                                            ? 'sticky left-0 z-10 bg-white' 
                                                            : ''
                                                    }`}
                                                    style={{
                                                        position: config.sticky ? 'sticky' : 'static',
                                                        left: config.sticky ? 0 : 'auto',
                                                        zIndex: config.sticky ? 10 : 'auto'
                                                    }}
                                                >
                                                    {config.renderCell(paper)}
                                                </td>
                                            ))}
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
} 