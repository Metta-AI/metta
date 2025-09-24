"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

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

  if (pathname.includes("/me")) return "me";
  return "feed"; // Default to feed view for root path
}

export function LibrarySidebar() {
  const pathname = usePathname();
  const activeNav = getActiveNavFromPath(pathname);

  return (
    <div className="fixed top-0 left-0 z-10 flex h-full w-48 flex-col border-r border-gray-200 bg-white">
      {/* Header Section */}
      <div className="border-b border-gray-100 px-6 py-6">
        <div className="flex items-center gap-3">
          {/* Bookshelf Icon */}
          <svg
            className="h-6 w-6 text-gray-500"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            suppressHydrationWarning
          >
            <path d="M3 18h18" />
            <rect x="3" y="8" width="3.6" height="10" />
            <rect x="6.6" y="6" width="3.6" height="12" />
            <rect x="10.2" y="9" width="3.6" height="9" />
            <rect x="13.8" y="7" width="3.6" height="11" />
            <rect x="17.4" y="5" width="3.6" height="13" />
          </svg>
          <h1 className="text-lg font-semibold text-gray-900">
            Republic of Papers
          </h1>
        </div>
      </div>

      {/* Navigation Section */}
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigationItems.map((item) => (
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
        ))}
      </nav>
    </div>
  );
}
