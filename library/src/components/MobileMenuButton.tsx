"use client";

import { Menu, X } from "lucide-react";
import { useMobileNav } from "./MobileNavProvider";

export function MobileMenuButton() {
  const { isSidebarOpen, toggleSidebar } = useMobileNav();

  return (
    <button
      onClick={toggleSidebar}
      className="flex h-8 w-8 items-center justify-center rounded-md text-gray-600 hover:bg-gray-100 hover:text-gray-900 md:hidden"
      aria-label={isSidebarOpen ? "Close menu" : "Open menu"}
    >
      {isSidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
    </button>
  );
}
