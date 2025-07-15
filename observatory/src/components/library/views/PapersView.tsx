/**
 * PapersView Component
 * 
 * Displays a comprehensive table of papers with sorting, filtering, and interactive features.
 * This component handles the papers view of the library, showing papers in a table format
 * with various metadata and user interactions.
 * 
 * The papers table includes:
 * - Paper titles with star/favorite functionality
 * - Author links to scholar profiles
 * - Affiliation links to institution profiles
 * - Research tags for categorization
 * - User avatars showing who has read or queued the paper
 * - External links to paper sources
 * - Star ratings and counts
 * 
 * Features:
 * - Sortable columns (title, stars)
 * - Interactive star/favorite toggling
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

import React, { useState } from 'react';
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
    onToggleStar
}: PapersViewProps) {
    
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
     * Handles sorting when a column header is clicked
     * 
     * When a user clicks a sortable column header:
     * - If it's the same column, we toggle the sort direction
     * - If it's a different column, we switch to that column and set ascending order
     * 
     * @param col - The column name to sort by (e.g., 'title', 'stars')
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
     * Renders a user avatar with hover functionality
     * 
     * @param user - The user object to display
     * @param bgColor - Background color class for the avatar
     * @param textColor - Text color class for the avatar
     * @returns JSX element for the user avatar
     */
    const renderUserAvatar = (user: any, bgColor: string, textColor: string) => (
        <span
            key={user.id}
            className={`inline-flex items-center justify-center w-6 h-6 rounded-full ${bgColor} ${textColor} text-xs font-bold border-2 border-white cursor-pointer`}
            title={user.name}
            onMouseEnter={(e) => handleUserHover(e, user)}
            onMouseLeave={handleUserLeave}
        >
            {user.avatar}
        </span>
    );

    return (
        <div className="p-6">
            <div className="max-w-6xl mx-auto">
                {/* Page Header */}
                <div className="mb-6">
                    <h1 className="text-2xl font-bold text-gray-900 mb-2">Papers</h1>
                    <p className="text-gray-600">Browse and sort all papers in the system</p>
                </div>
                
                {/* Papers Table */}
                <div className="overflow-x-auto relative">
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
                                    className="px-4 py-2 text-left cursor-pointer sticky left-0 z-10 bg-white border-r border-gray-200" 
                                    onClick={() => handleSort('title')}
                                >
                                    Title
                                    <span className="ml-1 align-middle">
                                        {papersSort.col === 'title' ? (papersSort.dir === 'asc' ? '▲' : '▼') : ''}
                                    </span>
                                </th>
                                
                                {/* Non-sortable columns */}
                                <th className="px-4 py-2 text-left">Authors</th>
                                <th className="px-4 py-2 text-left">Affiliations</th>
                                <th className="px-4 py-2 text-left">Tags</th>
                                <th className="px-4 py-2 text-left">Read by</th>
                                <th className="px-4 py-2 text-left">Link</th>
                                <th className="px-4 py-2 text-left">Queued</th>
                                
                                {/* Stars Column - sortable */}
                                <th 
                                    className="px-4 py-2 text-left cursor-pointer" 
                                    onClick={() => handleSort('stars')}
                                >
                                    Stars
                                    <span className="ml-1 align-middle">
                                        {papersSort.col === 'stars' ? (papersSort.dir === 'asc' ? '▲' : '▼') : ''}
                                    </span>
                                </th>
                            </tr>
                        </thead>
                        
                        <tbody>
                            {papers.map(paper => (
                                <tr key={paper.id} className="border-t border-gray-100 hover:bg-gray-50">
                                    {/* Title Cell - sticky left with star button */}
                                    <td className="px-4 py-2 whitespace-nowrap flex items-center gap-2 sticky left-0 z-10 bg-white border-r border-gray-200">
                                        <button 
                                            onClick={() => onToggleStar(paper.id)} 
                                            className="focus:outline-none"
                                            aria-label={paper.starred ? 'Remove from favorites' : 'Add to favorites'}
                                        >
                                            {paper.starred ? (
                                                <svg className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                                                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z"/>
                                                </svg>
                                            ) : (
                                                <svg className="w-4 h-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 20 20">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 15l-5.878 3.09 1.122-6.545L.488 6.91l6.564-.955L10 0l2.948 5.955 6.564.955-4.756 4.635 1.122 6.545z"/>
                                                </svg>
                                            )}
                                        </button>
                                        <span>{paper.title}</span>
                                    </td>
                                    
                                    {/* Authors Cell - links to scholar profiles */}
                                    <td className="px-4 py-2 whitespace-nowrap">
                                        {paper.authors.map((authorId: string, idx: number) => {
                                            const author = scholars.find(s => s.id === authorId);
                                            return author ? (
                                                <a 
                                                    key={author.id} 
                                                    href={`/scholars/${author.id}`} 
                                                    className="text-primary-600 hover:underline mr-1"
                                                >
                                                    {author.name}{idx < paper.authors.length - 1 ? ',' : ''}
                                                </a>
                                            ) : null;
                                        })}
                                    </td>
                                    
                                    {/* Affiliations Cell - links to affiliation profiles */}
                                    <td className="px-4 py-2 whitespace-nowrap">
                                        {paper.affiliations.map((affId: string, idx: number) => {
                                            const aff = affiliations.find(a => a.id === affId);
                                            return aff ? (
                                                <a 
                                                    key={aff.id} 
                                                    href={`/affiliations/${aff.id}`} 
                                                    className="text-primary-600 hover:underline mr-1"
                                                >
                                                    {aff.label}{idx < paper.affiliations.length - 1 ? ',' : ''}
                                                </a>
                                            ) : null;
                                        })}
                                    </td>
                                    
                                    {/* Tags Cell - research area tags */}
                                    <td className="px-4 py-2 whitespace-nowrap">
                                        {paper.tags.map((tag, idx) => (
                                            <span 
                                                key={idx} 
                                                className="inline-block bg-gray-100 text-gray-700 text-xs rounded-full px-2 py-0.5 mr-1 mb-0.5"
                                            >
                                                {tag}
                                            </span>
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
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
} 