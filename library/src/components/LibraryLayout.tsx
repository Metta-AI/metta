import { ReactNode } from 'react';
import { LibrarySidebar } from './LibrarySidebar';

/**
 * Library Layout Component
 * 
 * This component provides the main layout structure for the Library app,
 * combining the left sidebar with the main content area. The sidebar is
 * fixed positioned, so the main content is offset by the sidebar width.
 */

interface LibraryLayoutProps {
    children: ReactNode;
}

export function LibraryLayout({ children }: LibraryLayoutProps) {
    return (
        <div className="min-h-screen bg-gray-50">
            <div className="flex min-h-screen">
                {/* Left Sidebar */}
                <LibrarySidebar />
                
                {/* Main Content Area */}
                <div className="flex-1 ml-48 max-w-full overflow-hidden">
                    {children}
                </div>
            </div>
        </div>
    );
} 