"use client";

import {
  createContext,
  useContext,
  useState,
  ReactNode,
  useCallback,
} from "react";

interface MobileNavContextType {
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  closeSidebar: () => void;
}

const MobileNavContext = createContext<MobileNavContextType | undefined>(
  undefined
);

export function MobileNavProvider({ children }: { children: ReactNode }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const toggleSidebar = useCallback(() => {
    setIsSidebarOpen((prev) => !prev);
  }, [isSidebarOpen]);

  const closeSidebar = useCallback(() => {
    setIsSidebarOpen(false);
  }, []);

  return (
    <MobileNavContext.Provider
      value={{ isSidebarOpen, toggleSidebar, closeSidebar }}
    >
      {children}
    </MobileNavContext.Provider>
  );
}

export function useMobileNav() {
  const context = useContext(MobileNavContext);
  if (context === undefined) {
    throw new Error("useMobileNav must be used within a MobileNavProvider");
  }
  return context;
}
