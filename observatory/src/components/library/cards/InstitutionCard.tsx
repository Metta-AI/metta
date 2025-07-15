/**
 * InstitutionCard Component
 * 
 * Displays an individual institution/institution in a card format.
 * The card shows basic institution information and allows navigation to the institution's profile page.
 * 
 * Features:
 * - Click navigation to institution profile page
 * - Display of institution metrics (members, papers, citations)
 * - Tag display for research areas
 * - Admin actions for privileged users
 * - External website link
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';

interface InstitutionCardProps {
    institution: any;
    isAdmin: boolean;
    searchQuery?: string;
    onToggleFavorite?: (institutionId: string) => void;
    /**
     * Optional click handler for the card. If provided, called when the card is clicked.
     */
    onCardClick?: () => void;
}

export function InstitutionCard({ institution, isAdmin, searchQuery = '', onToggleFavorite, onCardClick }: InstitutionCardProps) {
    const navigate = useNavigate();

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
            className="w-full min-w-[20rem] max-w-full bg-white rounded-lg border border-gray-200 p-4 hover:shadow transition-shadow overflow-hidden flex flex-col cursor-pointer"
            onClick={onCardClick}
        >
            <div className="flex items-start justify-between mb-2 min-w-0">
                <div className="flex items-center gap-3 min-w-0">
                    {institution.logo ? (
                        <img src={institution.logo} alt={institution.label} className="w-12 h-12 rounded-full object-cover flex-shrink-0" />
                    ) : (
                        <div className="w-12 h-12 bg-primary-500 text-white rounded-full flex items-center justify-center text-lg font-semibold flex-shrink-0">
                            {institution.initials}
                        </div>
                    )}
                    <div className="min-w-0">
                        <h3 className="text-lg font-semibold text-gray-900 break-words leading-tight">{highlightMatchingText(institution.label, searchQuery)}</h3>
                        <p className="text-gray-600 text-sm break-words leading-tight">{highlightMatchingText(institution.name, searchQuery)}</p>
                        <p className="text-gray-500 text-xs break-words leading-tight">{highlightMatchingText(institution.location, searchQuery)}</p>
                    </div>
                </div>
                <div className="flex flex-col items-end gap-2 min-w-0">
                    <span className="px-3 py-0.5 rounded-full text-xs font-semibold mb-1 bg-gray-100 text-gray-600 border border-gray-200" title={institution.type}>{institution.type}</span>
                </div>
            </div>
            <div className="flex flex-wrap gap-1 mb-2">
                {institution.tags.map((tag: string, idx: number) => (
                    <span key={idx} className="px-2 py-0.5 bg-gray-100 text-gray-700 text-xs rounded-full">{highlightMatchingText(tag, searchQuery)}</span>
                ))}
            </div>
            <div className="flex items-center gap-4 text-xs text-gray-600 mb-2">
                <div>
                    <span className="font-semibold text-gray-900">{institution.memberCount}</span>
                    <span className="ml-1">members</span>
                </div>
                <div>
                    <span className="font-semibold text-gray-900">{institution.papers}</span>
                    <span className="ml-1">papers</span>
                </div>
                <div>
                    <span className="font-semibold text-gray-900">{institution.citations.toLocaleString()}</span>
                    <span className="ml-1">citations</span>
                </div>
            </div>
            <div className="border-t border-gray-200 pt-2 mt-2 flex items-center justify-between">
                <span className="text-xs text-gray-500">Active {institution.lastActive}</span>
                <div className="flex gap-2 items-center">
                    <a 
                        href={institution.website} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="text-xs text-primary-500 hover:text-primary-600 underline"
                        onClick={(e) => e.stopPropagation()}
                    >
                        Website
                    </a>
                    {isAdmin && (
                        <>
                            <button 
                                className="px-2 py-0.5 rounded bg-gray-100 text-gray-700 text-xs border border-gray-200 hover:bg-gray-200"
                                onClick={(e) => e.stopPropagation()}
                            >
                                Merge
                            </button>
                            <button 
                                className="px-2 py-0.5 rounded bg-gray-100 text-gray-700 text-xs border border-gray-200 hover:bg-gray-200"
                                onClick={(e) => e.stopPropagation()}
                            >
                                Mark Duplicate
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
} 