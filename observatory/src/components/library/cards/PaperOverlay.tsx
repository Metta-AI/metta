/**
 * PaperOverlay Component
 * 
 * Displays a comprehensive overlay for a paper with all available details
 * and interactive functionality including star/unstar, queue management,
 * and clickable authors/institutions that trigger their respective overlays.
 * 
 * Features:
 * - Clean, organized layout of all paper fields
 * - Clickable authors that trigger author overlays
 * - Clickable institutions that trigger institution overlays
 * - Full clickable URL with external link icon
 * - Star/unstar functionality
 * - Add to reading queue functionality
 * - Responsive design with proper spacing
 * 
 * Usage Example:
 * ```tsx
 * <PaperOverlay
 *   paper={paper}
 *   scholars={scholars}
 *   institutions={institutions}
 *   onToggleStar={handleToggleStar}
 *   onToggleQueue={handleToggleQueue}
 *   onShowAuthorOverlay={handleShowAuthorOverlay}
 *   onShowInstitutionOverlay={handleShowInstitutionOverlay}
 *   onClose={handleClose}
 * />
 * ```
 */

import React from 'react';

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
 * Props that the PaperOverlay component accepts
 */
interface PaperOverlayProps {
    /** The paper object to display */
    paper: Paper;
    
    /** Array of scholars for resolving author IDs to names */
    scholars: Scholar[];
    
    /** Array of institutions for resolving institution IDs to labels */
    institutions: Institution[];
    
    /** Callback function called when user toggles the star/favorite status */
    onToggleStar: (paperId: string) => void;
    
    /** Callback function called when user toggles queue status */
    onToggleQueue: (paperId: string) => void;
    
    /** Callback function called when an author should be shown in overlay */
    onShowAuthorOverlay: (authorId: string) => void;
    
    /** Callback function called when an institution should be shown in overlay */
    onShowInstitutionOverlay: (institutionId: string) => void;
    
    /** Callback function called when the overlay should be closed */
    onClose: () => void;
}

/**
 * PaperOverlay Component
 * 
 * Renders a comprehensive overlay displaying all paper details with
 * interactive functionality for star/unstar, queue management, and
 * clickable authors/institutions.
 */
export function PaperOverlay({
    paper,
    scholars,
    institutions,
    onToggleStar,
    onToggleQueue,
    onShowAuthorOverlay,
    onShowInstitutionOverlay,
    onClose
}: PaperOverlayProps) {
    
    // Validate required props
    if (!paper || typeof paper !== 'object' || !paper.id) {
        console.error('PaperOverlay: Invalid paper object provided');
        return null;
    }
    
    if (!Array.isArray(scholars)) {
        console.error('PaperOverlay: scholars prop must be an array');
        return null;
    }
    
    if (!Array.isArray(institutions)) {
        console.error('PaperOverlay: institutions prop must be an array');
        return null;
    }
    
    if (typeof onToggleStar !== 'function') {
        console.error('PaperOverlay: onToggleStar prop must be a function');
        return null;
    }
    
    if (typeof onToggleQueue !== 'function') {
        console.error('PaperOverlay: onToggleQueue prop must be a function');
        return null;
    }
    
    if (typeof onClose !== 'function') {
        console.error('PaperOverlay: onClose prop must be a function');
        return null;
    }
    
    /**
     * Handle click outside overlay to close
     */
    const handleBackdropClick = (e: React.MouseEvent) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };
    
    /**
     * Handle escape key to close overlay
     */
    React.useEffect(() => {
        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };
        
        document.addEventListener('keydown', handleEscape);
        return () => document.removeEventListener('keydown', handleEscape);
    }, [onClose]);
    
    /**
     * Check if paper is in user's queue
     */
    const isInQueue = Array.isArray(paper.queued) && paper.queued.length > 0;
    
    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={handleBackdropClick}
        >
            <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="flex items-start justify-between p-6 border-b border-gray-200">
                    <div className="flex-1 pr-4">
                        <h2 className="text-xl font-semibold text-gray-900 mb-2">
                            {paper.link && isValidUrl(paper.link) ? (
                                <a
                                    href={paper.link}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-gray-900 hover:text-primary-600 hover:underline transition-colors"
                                >
                                    {paper.title}
                                </a>
                            ) : (
                                paper.title
                            )}
                        </h2>
                        <div className="flex items-center gap-4 text-sm text-gray-600">
                            {paper.year && (
                                <span>Published: {paper.year}</span>
                            )}
                            {paper.citations !== undefined && (
                                <span>Citations: {paper.citations}</span>
                            )}
                            {paper.stars > 0 && (
                                <div className="flex items-center gap-1">
                                    <svg className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z"/>
                                    </svg>
                                    <span>{paper.stars}</span>
                                </div>
                            )}
                        </div>
                    </div>
                    
                    {/* Action Buttons */}
                    <div className="flex items-center gap-2">
                        {/* Star/Unstar Button */}
                        <button
                            onClick={() => onToggleStar(paper.id)}
                            className="p-2 rounded-lg hover:bg-gray-100 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500"
                            aria-label={paper.starred ? 'Remove from favorites' : 'Add to favorites'}
                        >
                            {paper.starred ? (
                                <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z"/>
                                </svg>
                            ) : (
                                <svg className="w-5 h-5 text-gray-300 hover:text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 20 20">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 15l-5.878 3.09 1.122-6.545L.488 6.91l6.564-.955L10 0l2.948 5.955 6.564.955-4.756 4.635 1.122 6.545z"/>
                                </svg>
                            )}
                        </button>
                        
                        {/* Queue Button */}
                        <button
                            onClick={() => onToggleQueue(paper.id)}
                            className={`p-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                                isInQueue 
                                    ? 'bg-primary-100 text-primary-700 hover:bg-primary-200' 
                                    : 'hover:bg-gray-100 text-gray-600'
                            }`}
                            aria-label={isInQueue ? 'Remove from queue' : 'Add to queue'}
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
                            </svg>
                        </button>
                        
                        {/* Close Button */}
                        <button
                            onClick={onClose}
                            className="p-2 rounded-lg hover:bg-gray-100 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500"
                            aria-label="Close overlay"
                        >
                            <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"/>
                            </svg>
                        </button>
                    </div>
                </div>
                
                {/* Content */}
                <div className="p-6 space-y-6">
                    {/* Authors */}
                    {paper.authors.length > 0 && (
                        <div>
                            <h3 className="text-sm font-medium text-gray-700 mb-2">Authors</h3>
                            <div className="flex flex-wrap gap-2">
                                {paper.authors.map((authorId: string) => {
                                    const author = scholars.find(s => s.id === authorId);
                                    if (!author) {
                                        return (
                                            <span key={`unknown-${authorId}`} className="text-gray-400 italic">
                                                Unknown Author
                                            </span>
                                        );
                                    }
                                    return (
                                        <button
                                            key={author.id}
                                            onClick={() => onShowAuthorOverlay(author.id)}
                                            className="text-primary-600 hover:text-primary-700 hover:underline bg-primary-50 hover:bg-primary-100 px-3 py-1 rounded-full text-sm transition-colors"
                                        >
                                            {author.name}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                    
                    {/* Institutions */}
                    {paper.institutions.length > 0 && (
                        <div>
                            <h3 className="text-sm font-medium text-gray-700 mb-2">Institutions</h3>
                            <div className="flex flex-wrap gap-2">
                                {paper.institutions.map((affId: string) => {
                                    const aff = institutions.find(a => a.id === affId);
                                    if (!aff) {
                                        return (
                                            <span key={`unknown-${affId}`} className="text-gray-400 italic">
                                                Unknown Institution
                                            </span>
                                        );
                                    }
                                    return (
                                        <button
                                            key={aff.id}
                                            onClick={() => onShowInstitutionOverlay(aff.id)}
                                            className="text-primary-600 hover:text-primary-700 hover:underline bg-primary-50 hover:bg-primary-100 px-3 py-1 rounded-full text-sm transition-colors"
                                        >
                                            {aff.label}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                    
                    {/* Tags */}
                    {paper.tags.length > 0 && (
                        <div>
                            <h3 className="text-sm font-medium text-gray-700 mb-2">Research Areas</h3>
                            <div className="flex flex-wrap gap-2">
                                {paper.tags.map((tag, idx) => (
                                    <span
                                        key={idx}
                                        className="bg-gray-100 text-gray-700 px-3 py-1 rounded-full text-sm"
                                    >
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                    
                    {/* Read By */}
                    {Array.isArray(paper.readBy) && paper.readBy.length > 0 && (
                        <div>
                            <h3 className="text-sm font-medium text-gray-700 mb-2">Read By</h3>
                            <div className="flex flex-wrap gap-2">
                                {paper.readBy.map((user: any) => (
                                    <div
                                        key={user.id || user.name}
                                        className="flex items-center gap-2 bg-primary-50 px-3 py-1 rounded-full"
                                    >
                                        <div className="w-6 h-6 rounded-full bg-primary-200 text-primary-800 text-xs font-medium flex items-center justify-center">
                                            {user.avatar || user.name.split(' ').map((n: string) => n[0]).join('').toUpperCase().slice(0, 2)}
                                        </div>
                                        <span className="text-sm text-gray-700">{user.name}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    
                    {/* Queued By */}
                    {Array.isArray(paper.queued) && paper.queued.length > 0 && (
                        <div>
                            <h3 className="text-sm font-medium text-gray-700 mb-2">Queued By</h3>
                            <div className="flex flex-wrap gap-2">
                                {paper.queued.map((user: any) => (
                                    <div
                                        key={user.id || user.name}
                                        className="flex items-center gap-2 bg-primary-50 px-3 py-1 rounded-full"
                                    >
                                        <div className="w-6 h-6 rounded-full bg-primary-100 text-primary-700 text-xs font-medium flex items-center justify-center">
                                            {user.avatar || user.name.split(' ').map((n: string) => n[0]).join('').toUpperCase().slice(0, 2)}
                                        </div>
                                        <span className="text-sm text-gray-700">{user.name}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    

                </div>
            </div>
        </div>
    );
} 