/**
 * PaperCard Component
 * 
 * Displays an individual paper in a card format with expandable abstract functionality.
 * The card shows paper metadata and allows users to expand/collapse the abstract text.
 * 
 * Features:
 * - Expandable abstract text
 * - Paper metadata display (title, authors, year, citations)
 * - External link to paper
 * - MathJax support for mathematical content
 * - Responsive design
 */

import React from 'react';

interface PaperCardProps {
    paper: any;
    expandedAbstracts: Set<string>;
    onToggleAbstract: (paperId: string) => void;
}

export function PaperCard({ paper, expandedAbstracts, onToggleAbstract }: PaperCardProps) {
    const isExpanded = expandedAbstracts.has(paper.id);

    return (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-start justify-between mb-3">
                <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2 leading-tight">
                        <a 
                            href={paper.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="hover:text-primary-600 transition-colors"
                        >
                            {paper.title}
                        </a>
                    </h3>
                    <p className="text-sm text-gray-600 mb-2">
                        {(paper.authors ?? []).join(', ')}
                    </p>
                    <div className="flex items-center gap-4 text-xs text-gray-500 mb-3">
                        <span>{paper.year}</span>
                        <span>•</span>
                        <span>{paper.citations} citations</span>
                        <span>•</span>
                        <span>{(paper.affiliations ?? []).join(', ')}</span>
                    </div>
                </div>
                <div className="flex items-center gap-2 ml-4">
                    <button
                        onClick={() => onToggleAbstract(paper.id)}
                        className="text-xs text-primary-600 hover:text-primary-700 font-medium"
                    >
                        {isExpanded ? 'Hide Abstract' : 'Show Abstract'}
                    </button>
                </div>
            </div>
            
            {isExpanded && paper.abstract && (
                <div className="border-t border-gray-100 pt-3">
                    <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                        {paper.abstract}
                    </p>
                </div>
            )}
            
            <div className="flex flex-wrap gap-1 mt-3">
                {(paper.tags ?? []).map((tag: string, index: number) => (
                    <span 
                        key={index} 
                        className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full"
                    >
                        {tag}
                    </span>
                ))}
            </div>
        </div>
    );
} 