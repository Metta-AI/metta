"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { signOut } from "next-auth/react";
import { useMobileNav } from "./MobileNavProvider";
import { useEffect, useRef } from "react";

/**
 * Library Sidebar Component
 *
 * This component recreates the left-hand navigation sidebar from the original observatory app.
 * It provides navigation between different views: Feed, Papers, Authors, Institutions, Groups, and Me.
 * Uses simplified gray outline icons as per user preferences.
 */

// Navigation items configuration with simplified gray outline icons
const navigationItems = [
  {
    id: "feed",
    label: "Feed",
    href: "/",
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        suppressHydrationWarning
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M4 6h16M4 12h16M4 18h16"
        />
      </svg>
    ),
  },
  {
    id: "papers",
    label: "Papers",
    href: "/papers",
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        suppressHydrationWarning
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
      </svg>
    ),
  },
  {
    id: "authors",
    label: "Authors",
    href: "/authors",
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        suppressHydrationWarning
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
        />
      </svg>
    ),
  },
  {
    id: "institutions",
    label: "Institutions",
    href: "/institutions",
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        suppressHydrationWarning
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 4h1m-1 4h1m4-4h1m-1 4h1"
        />
      </svg>
    ),
  },
  {
    id: "groups",
    label: "Groups",
    href: "/groups",
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        suppressHydrationWarning
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z"
        />
      </svg>
    ),
  },
  {
    id: "me",
    label: "Me",
    href: "/me",
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        suppressHydrationWarning
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
        />
      </svg>
    ),
  },
  {
    id: "settings",
    label: "Settings",
    href: "/settings",
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        suppressHydrationWarning
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
        />
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
        />
      </svg>
    ),
  },
  {
    id: "signout",
    label: "Sign Out",
    href: "#", // We'll handle this specially
    icon: (
      <svg
        className="h-5 w-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        suppressHydrationWarning
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
        />
      </svg>
    ),
  },
];

/**
 * Determines the active navigation item based on the current URL path
 *
 * @param pathname - The current URL pathname
 * @returns The ID of the active navigation item
 */
function getActiveNavFromPath(pathname: string): string {
  if (pathname.includes("/authors")) return "authors";
  if (pathname.includes("/institutions")) return "institutions";
  if (pathname.includes("/groups")) return "groups";
  if (pathname.includes("/papers")) return "papers";
  if (pathname.includes("/settings")) return "settings";
  if (pathname.includes("/me")) return "me";
  return "feed"; // Default to feed view for root path
}

export function LibrarySidebar() {
  const pathname = usePathname();
  const activeNav = getActiveNavFromPath(pathname);
  const { isSidebarOpen, closeSidebar } = useMobileNav();

  // Close sidebar when navigating on mobile
  const prevPathname = useRef(pathname);
  useEffect(() => {
    if (prevPathname.current !== pathname) {
      closeSidebar();
      prevPathname.current = pathname;
    }
  }, [pathname, closeSidebar]);

  return (
    <>
      {/* Mobile backdrop - transparent but clickable */}
      {isSidebarOpen && (
        <div className="inset-0 z-40 md:hidden" onClick={closeSidebar} />
      )}

      {/* Sidebar */}
      <div
        className={`z-50 flex h-full w-48 flex-col border-r border-gray-200 bg-white transition-transform duration-300 ease-in-out md:z-10 md:translate-x-0 ${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        {/* Header Section - removed, logo moved to TopMenu */}

        {/* Navigation Section */}
        <nav className="flex-1 space-y-1 px-3 py-4">
          {navigationItems.map((item) => {
            // Handle Sign Out specially
            if (item.id === "signout") {
              return (
                <button
                  key={item.id}
                  onClick={() => signOut()}
                  className="mx-1 flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2.5 text-left text-gray-600 transition-colors hover:bg-gray-50 hover:text-gray-900"
                >
                  {item.icon}
                  <span className="text-sm font-medium">{item.label}</span>
                </button>
              );
            }

            // Regular navigation items
            return (
              <Link
                key={item.id}
                href={item.href}
                className={`mx-1 flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors ${
                  activeNav === item.id
                    ? "bg-primary-50 text-primary-700 border-primary-200 border"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                }`}
              >
                {item.icon}
                <span className="text-sm font-medium">{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </>
  );
}
