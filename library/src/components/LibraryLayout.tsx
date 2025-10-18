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
      <div className="flex h-screen w-full flex-col bg-gray-50">
        {/* Top Navigation Bar */}
        <div className="z-30">
          <TopMenu />
        </div>

        <div className="flex h-full w-full overflow-hidden">
          <div className="flex h-full w-0 md:w-48">
            <LibrarySidebar />
          </div>

          {/* Main Layout Area */}
          <div className="flex w-full flex-1 overflow-hidden">
            {/* Main Content Area */}
            <div className="h-100vh flex w-full flex-1 items-start overflow-x-hidden overflow-y-auto">
              {children}
            </div>
          </div>
        </div>
      </div>
    </MobileNavProvider>
  );
}
