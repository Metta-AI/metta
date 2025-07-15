/**
 * AuthorCard Component
 * 
 * Displays an individual author in a card format with hover expansion functionality.
 * The card shows basic author information and expands on hover to show additional details
 * like recent papers and metrics. Clicking the card navigates to the author's profile page.
 * 
 * Features:
 * - Hover expansion with additional author details
 * - Follow/unfollow functionality for claimed profiles
 * - Click navigation to author profile
 * - Expertise tags that can be clicked to filter
 * - Responsive design with proper text truncation
 */

import React, { useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

interface AuthorCardProps {
    author: any;
    expanded: boolean;
    onExpand: (id: string) => void;
    onCollapse: (id: string) => void;
    searchQuery: string;
    onToggleFollow: (authorId: string) => void;
    onTagClick: (tag: string) => void;
    filterInputRef: React.RefObject<HTMLInputElement | null>;
    /**
     * Optional click handler for the card. If provided, called when the card is clicked.
     */
    onCardClick?: () => void;
}

export function AuthorCard({ 
    author, 
    expanded, 
    onExpand, 
    onCollapse, 
    searchQuery, 
    onToggleFollow, 
    onTagClick,
    filterInputRef,
    onCardClick
}: AuthorCardProps) {
    const cardRef = useRef<HTMLDivElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const tagsToShow = author.expertise;

    // Helper function to highlight matching text in search results
    const highlightMatchingText = (text: string, query: string) => {
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
        <div
            ref={containerRef}
            className="relative break-inside-avoid mb-6"
        >
            <div 
                ref={cardRef}
                className="relative w-full bg-white rounded-lg border border-gray-200 p-3 hover:shadow-md transition-all cursor-pointer min-h-[11rem] h-44 flex flex-col"
                onClick={onCardClick}
            >
                {/* Avatar and follow button section */}
                <div className="flex items-start gap-3 min-w-0 mb-3 min-h-[3.5rem] h-16 flex-shrink-0">
                    <div className="flex flex-col items-center flex-shrink-0" style={{ width: 60 }}>
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center text-sm font-semibold ${author.claimed ? 'bg-primary-500 text-white' : 'bg-gray-300 text-gray-600'}`}> 
                            {author.initials || (author.name.split(' ').map((n: string) => n[0]).join('').toUpperCase())}
                        </div>
                        {author.claimed && (
                            <button
                                onClick={e => { e.stopPropagation(); onToggleFollow(author.id); }}
                                className={`mt-1 px-2 py-0.5 rounded-full text-[9px] uppercase tracking-wider font-semibold transition-colors ${
                                    author.isFollowing
                                        ? 'bg-orange-100 text-orange-700'
                                        : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                                }`}
                            >
                                {author.isFollowing ? 'FOLLOWING' : 'FOLLOW'}
                            </button>
                        )}
                    </div>
                    <div className="flex-1 min-w-0">
                        <h3 className="text-base font-semibold text-gray-900 break-words leading-tight mb-1">
                            {highlightMatchingText(author.name, searchQuery)}
                        </h3>
                        <p className="text-gray-600 text-sm break-words leading-tight">
                            {highlightMatchingText(author.institution, searchQuery)}
                        </p>
                    </div>
                </div>
                {/* Expertise tags as flowing text with consistent spacing */}
                <div className="w-full">
                    <div className="text-xs font-semibold text-gray-700 leading-tight break-words">
                        {tagsToShow.map((exp: string, index: number) => (
                            <React.Fragment key={exp}>
                                {index > 0 && <span className="text-gray-400"> • </span>}
                                <button
                                    type="button"
                                    className="hover:text-primary-600 hover:underline transition-colors cursor-pointer p-0 m-0 bg-transparent border-none font-semibold break-words"
                                    style={{ display: 'inline', background: 'none' }}
                                    onClick={e => {
                                        e.stopPropagation();
                                        onTagClick(exp);
                                        filterInputRef.current?.focus();
                                    }}
                                >
                                    {highlightMatchingText(exp, searchQuery)}
                                </button>
                            </React.Fragment>
                        ))}
                    </div>
                </div>
                {/* Author metrics row at the bottom */}
                <div className="flex items-center justify-between mt-auto pt-3 border-t border-gray-100 text-xs text-gray-600">
                    <div><span className="font-semibold text-gray-900">{author.hIndex}</span> h-index</div>
                    <div><span className="font-semibold text-gray-900">{author.papers.length}</span> papers</div>
                    <div><span className="font-semibold text-gray-900">{author.totalCitations.toLocaleString()}</span> citations</div>
                </div>
            </div>
        </div>
    );
} 