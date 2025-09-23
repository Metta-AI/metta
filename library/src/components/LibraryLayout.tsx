import { ReactNode } from "react";
import { LibrarySidebar } from "./LibrarySidebar";
import { TopMenu } from "@/app/TopMenu";

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
      {/* Top Navigation Bar */}
      <TopMenu />

      <div className="flex min-h-screen">
        {/* Left Sidebar */}
        <LibrarySidebar />

        {/* Main Content Area */}
        <div className="ml-48 max-w-full flex-1 overflow-hidden">
          {children}
        </div>
      </div>
    </div>
  );
}
