'use client';

/**
 * Library Sidebar Component
 * 
 * This component recreates the left-hand sidebar from the original observatory app.
 * It includes the header with the bookshelf icon and navigation items for different views.
 * 
 * Features:
 * - Fixed positioning on the left side
 * - Bookshelf icon and "Library" title in header
 * - Navigation items with icons and labels
 * - Active state styling for selected items
 * - Hover effects and transitions
 * 
 * Navigation Items:
 * - Feed: Main feed view
 * - Papers: Papers table view
 * - Search: Search functionality
 * - Authors: Authors/people view
 * - Institutions: Institutions view
 * - Me: User profile view
 */

import React from 'react';

// Navigation items configuration matching the original design
const navItems = [
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
        id: 'authors',
        label: 'Authors',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
        )
    },
    {
        id: 'institutions',
        label: 'Institutions',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {/* Building with simplified design */}
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 4h1m-1 4h1m4-4h1m-1 4h1" />
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

interface LibrarySidebarProps {
    activeNav?: string;
    onNavClick?: (id: string) => void;
}

export function LibrarySidebar({ activeNav = 'feed', onNavClick }: LibrarySidebarProps) {
    const handleNavClick = (id: string) => {
        if (onNavClick) {
            onNavClick(id);
        }
        // For now, just log the click since we're not implementing functionality yet
        console.log('Navigation clicked:', id);
    };

    return (
        <div className="w-48 bg-white border-r border-gray-200 fixed top-0 left-0 h-full flex flex-col z-10">
            {/* Header Section */}
            <div className="px-6 py-6 border-b border-gray-100">
                <div className="flex items-center gap-3">
                    {/* Bookshelf Icon */}
                    <svg className="w-6 h-6 text-gray-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 18h18" />
                        <rect x="3" y="8" width="3.6" height="10" />
                        <rect x="6.6" y="6" width="3.6" height="12" />
                        <rect x="10.2" y="9" width="3.6" height="9" />
                        <rect x="13.8" y="7" width="3.6" height="11" />
                        <rect x="17.4" y="5" width="3.6" height="13" />
                    </svg>
                    <h1 className="text-lg font-semibold text-gray-900">Library</h1>
                </div>
            </div>

            {/* Navigation Section */}
            <nav className="flex-1 px-3 py-4 space-y-1">
                {navItems.map(item => (
                    <button
                        key={item.id}
                        onClick={() => handleNavClick(item.id)}
                        className={`w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors rounded-lg mx-1 ${
                            activeNav === item.id
                                ? 'bg-primary-50 text-primary-700 border border-primary-200'
                                : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                        }`}
                    >
                        {item.icon}
                        <span className="font-medium text-sm">{item.label}</span>
                    </button>
                ))}
            </nav>
        </div>
    );
} 