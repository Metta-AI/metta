/**
 * AffiliationCard Component
 * 
 * Displays an individual affiliation/institution in a card format.
 * The card shows basic affiliation information and allows navigation to the affiliation's profile page.
 * 
 * Features:
 * - Click navigation to affiliation profile page
 * - Display of affiliation metrics (members, papers, citations)
 * - Tag display for research areas
 * - Admin actions for privileged users
 * - External website link
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';

interface AffiliationCardProps {
    affiliation: any;
    isAdmin: boolean;
    onToggleFavorite?: (affiliationId: string) => void;
    /**
     * Optional click handler for the card. If provided, called when the card is clicked.
     */
    onCardClick?: () => void;
}

export function AffiliationCard({ affiliation, isAdmin, onToggleFavorite, onCardClick }: AffiliationCardProps) {
    const navigate = useNavigate();

    return (
        <div 
            className="w-full min-w-[20rem] max-w-full bg-white rounded-lg border border-gray-200 p-4 hover:shadow transition-shadow overflow-hidden flex flex-col cursor-pointer"
            onClick={onCardClick}
        >
            <div className="flex items-start justify-between mb-2 min-w-0">
                <div className="flex items-center gap-3 min-w-0">
                    {affiliation.logo ? (
                        <img src={affiliation.logo} alt={affiliation.label} className="w-12 h-12 rounded-full object-cover flex-shrink-0" />
                    ) : (
                        <div className="w-12 h-12 bg-primary-500 text-white rounded-full flex items-center justify-center text-lg font-semibold flex-shrink-0">
                            {affiliation.initials}
                        </div>
                    )}
                    <div className="min-w-0">
                        <h3 className="text-lg font-semibold text-gray-900 break-words leading-tight">{affiliation.label}</h3>
                        <p className="text-gray-600 text-sm break-words leading-tight">{affiliation.name}</p>
                        <p className="text-gray-500 text-xs break-words leading-tight">{affiliation.location}</p>
                    </div>
                </div>
                <div className="flex flex-col items-end gap-2 min-w-0">
                    <span className="px-3 py-0.5 rounded-full text-xs font-semibold mb-1 bg-gray-100 text-gray-600 border border-gray-200" title={affiliation.type}>{affiliation.type}</span>
                </div>
            </div>
            <div className="flex flex-wrap gap-1 mb-2">
                {affiliation.tags.map((tag: string, idx: number) => (
                    <span key={idx} className="px-2 py-0.5 bg-gray-100 text-gray-700 text-xs rounded-full">{tag}</span>
                ))}
            </div>
            <div className="flex items-center gap-4 text-xs text-gray-600 mb-2">
                <div>
                    <span className="font-semibold text-gray-900">{affiliation.memberCount}</span>
                    <span className="ml-1">members</span>
                </div>
                <div>
                    <span className="font-semibold text-gray-900">{affiliation.papers}</span>
                    <span className="ml-1">papers</span>
                </div>
                <div>
                    <span className="font-semibold text-gray-900">{affiliation.citations.toLocaleString()}</span>
                    <span className="ml-1">citations</span>
                </div>
            </div>
            <div className="border-t border-gray-200 pt-2 mt-2 flex items-center justify-between">
                <span className="text-xs text-gray-500">Active {affiliation.lastActive}</span>
                <div className="flex gap-2 items-center">
                    <a 
                        href={affiliation.website} 
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