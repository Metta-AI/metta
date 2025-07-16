/**
 * Library Utility Functions
 * 
 * This file contains all utility functions for the Library application.
 * These functions provide reusable logic for filtering, sorting, MathJax integration,
 * and navigation that can be used across different components.
 */

import React from 'react';
import { Scholar, Institution, Paper, FilterSortConfig } from './LibraryTypes';

// ============================================================================
// NAVIGATION UTILITIES
// ============================================================================

/**
 * Determines the active navigation item based on the current URL path
 * 
 * @param pathname - The current URL pathname
 * @returns The ID of the active navigation item
 * 
 * This function maps URL paths to navigation items, allowing the sidebar
 * to stay in sync with the browser's URL. It checks for path segments
 * in order of specificity (most specific first).
 */
export function getNavFromPath(pathname: string): string {
    if (pathname.includes('/institutions')) return 'institutions';
    if (pathname.includes('/papers')) return 'papers';
    if (pathname.includes('/me')) return 'me';
    if (pathname.includes('/search')) return 'search';
    return 'feed'; // Default to feed view
}

/**
 * Navigation items configuration for the Library sidebar
 * 
 * This array defines all available navigation items with their icons and labels.
 * Each item has a unique ID that corresponds to a view in the Library app.
 */
export const navigationItems = [
    {
        id: 'feed',
        label: 'Feed',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
        )
    },
    {
        id: 'papers',
        label: 'Papers',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
        )
    },
    {
        id: 'search',
        label: 'Search',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <circle cx="11" cy="11" r="8" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m21 21-4.35-4.35" />
            </svg>
        )
    },


    {
        id: 'institutions',
        label: 'Institutions',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
            </svg>
        )
    },
    {
        id: 'me',
        label: 'Me',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
        )
    }
];

// ============================================================================
// FILTERING AND SORTING UTILITIES
// ============================================================================

/**
 * Filters and sorts scholars based on search query and sort configuration
 * 
 * @param scholars - Array of all scholars
 * @param config - Filtering and sorting configuration
 * @returns Filtered and sorted array of scholars
 * 
 * This function applies search filtering (by name, institution, expertise)
 * and sorting (by various scholar attributes) to the scholars array.
 * Filtering only occurs when the query is at least 2 characters long.
 */
export function filterAndSortScholars(
    scholars: Scholar[], 
    config: FilterSortConfig
): Scholar[] {
    const { query, sortBy, sortDirection, minQueryLength = 2 } = config;
    
    return scholars
        .filter(scholar => {
            // Only filter when search query meets minimum length
            if (query.length < minQueryLength) return true;
            
            const searchQuery = query.toLowerCase();
            return scholar.name.toLowerCase().includes(searchQuery) ||
                scholar.institution.toLowerCase().includes(searchQuery) ||
                scholar.expertise.some(exp => exp.toLowerCase().includes(searchQuery));
        })
        .sort((a, b) => {
            let aValue: string | number;
            let bValue: string | number;
            
            switch (sortBy) {
                case 'name':
                    // Sort by last name
                    const aLastName = a.name.split(' ').pop()?.toLowerCase() || '';
                    const bLastName = b.name.split(' ').pop()?.toLowerCase() || '';
                    aValue = aLastName;
                    bValue = bLastName;
                    break;
                case 'institution':
                    aValue = a.institution.toLowerCase();
                    bValue = b.institution.toLowerCase();
                    break;
                case 'recentActivity':
                    // Convert activity strings to numbers for sorting (lower = more recent)
                    const activityOrder = { 
                        '1 day ago': 1, '2 days ago': 2, '3 days ago': 3, 
                        '4 days ago': 4, '5 days ago': 5, '1 week ago': 7 
                    };
                    aValue = activityOrder[a.recentActivity as keyof typeof activityOrder] || 10;
                    bValue = activityOrder[b.recentActivity as keyof typeof activityOrder] || 10;
                    break;
                case 'papers':
                    aValue = a.papers.length;
                    bValue = b.papers.length;
                    break;
                case 'citations':
                    aValue = a.totalCitations;
                    bValue = b.totalCitations;
                    break;
                case 'hIndex':
                    aValue = a.hIndex;
                    bValue = b.hIndex;
                    break;
                default:
                    return 0;
            }
            
            if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
            if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
}

/**
 * Filters and sorts institutions based on search query and sort configuration
 * 
 * @param institutions - Array of all institutions
 * @param config - Filtering and sorting configuration
 * @returns Filtered and sorted array of institutions
 * 
 * This function applies search filtering (by name, location, tags)
 * and sorting (by various institution attributes) to the institutions array.
 * Filtering only occurs when the query is at least 2 characters long.
 */
export function filterAndSortInstitutions(
    institutions: Institution[], 
    config: FilterSortConfig
): Institution[] {
    const { query, sortBy, sortDirection, minQueryLength = 2 } = config;
    
    return institutions
        .filter(institution => {
            // Only filter when search query meets minimum length
            if (query.length < minQueryLength) return true;
            
            const searchQuery = query.toLowerCase();
            return institution.label.toLowerCase().includes(searchQuery) ||
                institution.name.toLowerCase().includes(searchQuery) ||
                institution.location.toLowerCase().includes(searchQuery) ||
                institution.tags.some(tag => tag.toLowerCase().includes(searchQuery));
        })
        .sort((a, b) => {
            let aValue: string | number;
            let bValue: string | number;
            
            switch (sortBy) {
                case 'name':
                    aValue = a.label.toLowerCase();
                    bValue = b.label.toLowerCase();
                    break;
                case 'location':
                    aValue = a.location.toLowerCase();
                    bValue = b.location.toLowerCase();
                    break;
                case 'type':
                    aValue = a.type.toLowerCase();
                    bValue = b.type.toLowerCase();
                    break;
                case 'members':
                    aValue = a.memberCount;
                    bValue = b.memberCount;
                    break;
                case 'papers':
                    aValue = a.papers;
                    bValue = b.papers;
                    break;
                case 'citations':
                    aValue = a.citations;
                    bValue = b.citations;
                    break;
                default:
                    return 0;
            }
            
            if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
            if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
}

/**
 * Highlights matching text in a string based on a search query
 * 
 * @param text - The text to highlight
 * @param query - The search query to highlight
 * @returns React nodes with highlighted text
 * 
 * This function splits text by the search query and wraps matching parts
 * in a strong tag for highlighting. It's used to show search results
 * with visual emphasis on matching terms.
 */
export function highlightMatchingText(text: string, query: string): React.ReactNode {
    if (query.length < 2) return text;
    
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, index) => 
        regex.test(part) ? (
            <strong key={index} className="font-bold">{part}</strong>
        ) : (
            part
        )
    );
}

// ============================================================================
// MATHJAX INTEGRATION UTILITIES
// ============================================================================

/**
 * MathJax type declarations for global window object
 * 
 * This interface extends the global Window interface to include MathJax
 * configuration and methods for rendering mathematical content.
 */
declare global {
    interface Window {
        MathJax?: {
            tex: {
                inlineMath: string[][]
                displayMath: string[][]
                processEscapes: boolean
                processEnvironments: boolean
            }
            options: {
                skipHtmlTags: string[]
            }
            typesetPromise?: (elements: HTMLElement[]) => Promise<void>
        }
    }
}

/**
 * Loads MathJax library dynamically and configures it for the Library app
 * 
 * @returns Promise that resolves when MathJax is loaded and configured
 * 
 * This function dynamically loads the MathJax library from CDN and configures
 * it for rendering mathematical content in the feed. It sets up inline and
 * display math delimiters and configures which HTML tags to skip.
 */
export async function loadMathJax(): Promise<void> {
    if (typeof window !== 'undefined' && !window.MathJax) {
        // Configure MathJax before loading
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }
        };

        // Load MathJax dynamically
        const mathJaxScript = document.createElement('script');
        mathJaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
        mathJaxScript.async = true;
        
        return new Promise((resolve, reject) => {
            mathJaxScript.onload = () => {
                // Wait a bit for MathJax to fully initialize
                setTimeout(() => {
                    if (window.MathJax) {
                        resolve();
                    } else {
                        reject(new Error('MathJax failed to initialize'));
                    }
                }, 200);
            };
            mathJaxScript.onerror = () => reject(new Error('Failed to load MathJax'));
            document.head.appendChild(mathJaxScript);
        });
    } else if (window.MathJax) {
        return Promise.resolve();
    }
}

/**
 * Renders mathematical content in a DOM element using MathJax
 * 
 * @param element - The DOM element containing mathematical content
 * @returns Promise that resolves when rendering is complete
 * 
 * This function uses MathJax to render mathematical expressions in the
 * specified DOM element. It handles both inline and display math.
 */
export async function renderMathJax(element: HTMLElement): Promise<void> {
    if (window.MathJax?.typesetPromise) {
        try {
            await window.MathJax.typesetPromise([element]);
            console.log('MathJax rendering completed successfully');
        } catch (err: any) {
            console.error('MathJax error:', err);
            throw err;
        }
    }
}

// ============================================================================
// MOCK DATA GENERATION UTILITIES
// ============================================================================

/**
 * Generates mock posts for the feed view
 * 
 * @returns Array of mock post objects
 * 
 * This function creates sample posts for the feed, including user posts,
 * paper posts, and pure paper posts. It's used to populate the feed
 * with realistic-looking content for development and testing.
 */
export function generateMockPosts(): any[] {
    return [
        {
            id: 'post1',
            type: 'user-post',
            name: 'Alice',
            avatar: 'A',
            time: '2 hours ago',
            content: 'Just finished reading this fascinating paper on transformer architectures. The attention mechanism insights are really groundbreaking!',
            replies: 3,
            retweets: 12,
            likes: 45
        },
        {
            id: 'post2',
            type: 'paper-post',
            name: 'Bob',
            avatar: 'B',
            time: '4 hours ago',
            content: 'Check out this new research on attention mechanisms:',
            paper: {
                id: 'paper1',
                title: 'Attention Is All You Need',
                authors: ['Vaswani', 'Shazeer', 'Parmar'],
                year: 2017,
                citations: 45000,
                url: 'https://arxiv.org/abs/1706.03762'
            },
            replies: 8,
            retweets: 25,
            likes: 67
        },
        {
            id: 'post3',
            type: 'pure-paper',
            paper: {
                id: 'paper2',
                title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                authors: ['Devlin', 'Chang', 'Lee', 'Toutanova'],
                year: 2018,
                citations: 35000,
                url: 'https://arxiv.org/abs/1810.04805'
            }
        }
    ];
} 