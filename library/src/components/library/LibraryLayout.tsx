'use client';

/**
 * Library Layout Component
 * 
 * This component provides the main layout for the library application,
 * including the fixed sidebar and the main content area that adjusts
 * to accommodate the sidebar width.
 * 
 * Features:
 * - Fixed sidebar on the left (192px width)
 * - Main content area with proper left margin
 * - Full height layout
 * - Responsive design considerations
 */

import React from 'react';
import { LibrarySidebar } from './LibrarySidebar';

interface LibraryLayoutProps {
    children: React.ReactNode;
    activeNav?: string;
    onNavClick?: (id: string) => void;
}

export function LibraryLayout({ children, activeNav = 'feed', onNavClick }: LibraryLayoutProps) {
    return (
        <div className="min-h-screen bg-gray-50 font-inter">
            <div className="flex min-h-screen">
                {/* Left Sidebar */}
                <LibrarySidebar activeNav={activeNav} onNavClick={onNavClick} />
                
                {/* Main Content */}
                <div className="flex-1 ml-48 max-w-full overflow-hidden">
                    {children}
                </div>
            </div>
        </div>
    );
} 