import { ReactNode } from "react";
import { LibrarySidebar } from "./LibrarySidebar";
import { TopMenu } from "@/app/TopMenu";
import { MobileNavProvider } from "./MobileNavProvider";

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
    <MobileNavProvider>
      <div className="flex h-screen flex-col bg-gray-50">
        {/* Top Navigation Bar - Sticky */}
        <div className="sticky top-0 z-30">
          <TopMenu />
        </div>

        {/* Main Layout Area */}
        <div className="flex flex-1 overflow-hidden">
          {/* Left Sidebar */}
          <LibrarySidebar />

          {/* Main Content Area */}
          <div className="w-full flex-1 overflow-hidden md:ml-48">
            {children}
          </div>
        </div>
      </div>
    </MobileNavProvider>
  );
}
