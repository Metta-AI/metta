"use client";

import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  PaperWithUserContext,
  User,
  UserInteraction,
} from "@/posts/data/papers";

import { toggleQueueAction } from "@/posts/actions/toggleQueueAction";
import { useOverlayNavigation } from "./OverlayStack";
import UserCard from "./UserCard";
import { StarWidgetQuery } from "./StarWidgetQuery";
import { useStarMutation } from "@/hooks/useStarMutation";

/**
 * PapersView Component
 *
 * Displays a comprehensive table of papers with sorting, filtering, and interactive features.
 * This component uses native table semantics with sticky positioning for frozen columns,
 * providing robust cross-browser compatibility and proper accessibility.
 *
 * Features:
 * - Sortable columns (all columns are clickable to sort)
 * - Interactive star/favorite toggling
 * - Clickable tags that apply search filters
 * - User hover cards showing user details
 * - Frozen title column with horizontal scrolling for other columns
 * - All columns individually resizable with proper handles
 * - Links to related papers
 * - Hover tooltips for long titles
 */

interface PapersViewProps {
  papers: PaperWithUserContext[];
  users: User[];
  interactions: UserInteraction[];
  initialSearch?: string;
}

/**
 * Utility function to validate if a string is a valid URL
 */
const isValidUrl = (url: string): boolean => {
  if (!url || typeof url !== "string") {
    return false;
  }

  try {
    const urlObj = new URL(url);
    return urlObj.protocol === "http:" || urlObj.protocol === "https:";
  } catch {
    return false;
  }
};

/**
 * Column configuration for the table
 */
interface ColumnConfig {
  key: string;
  label: string;
  width: number;
  minWidth: number;
  maxWidth: number;
  sortable: boolean;
  sticky?: boolean; // For frozen column
  renderHeader: (sortIndicator: React.ReactNode) => React.ReactNode;
  renderCell: (paper: PaperWithUserContext) => React.ReactNode;
}

export function PapersView({
  papers,
  users,
  interactions,
  initialSearch = "",
}: PapersViewProps) {
  // State for search and filtering
  const [searchQuery, setSearchQuery] = useState(initialSearch);
  const [showOnlyStarred, setShowOnlyStarred] = useState(false);
  const [sortColumn, setSortColumn] = useState<
    "title" | "tags" | "readBy" | "queued"
  >("title");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  // State for column widths (all resizable)
  const [columnWidths, setColumnWidths] = useState({
    title: 400,
    tags: 300,
    readBy: 120,
    queued: 120,
  });

  // Drag state for column resizing
  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(0);
  const [mouseX, setMouseX] = useState(0);
  const draggedColumnRef = useRef<string | null>(null);
  const tableRef = useRef<HTMLTableElement>(null);

  // State for user card
  const [selectedUser, setSelectedUser] = useState<User | null>(null);

  // State for loading indicator during filtering/sorting operations
  const [isLoading, setIsLoading] = useState(false);

  // Router for URL navigation
  const router = useRouter();

  // Overlay navigation
  const { openPaper } = useOverlayNavigation();

  // Star mutation
  const starMutation = useStarMutation();

  // Create a map of users for quick lookup
  const usersMap = useMemo(() => {
    const map = new Map<string, User>();
    users.forEach((user) => {
      map.set(user.id, user);
    });
    return map;
  }, [users]);

  // Get users who have read a specific paper
  const getReadersForPaper = (paperId: string): User[] => {
    return interactions
      .filter(
        (interaction) => interaction.paperId === paperId && interaction.readAt
      )
      .map((interaction) => usersMap.get(interaction.userId))
      .filter((user): user is User => user !== undefined);
  };

  // Get users who have queued a specific paper
  const getQueuedForPaper = (paperId: string): User[] => {
    return interactions
      .filter(
        (interaction) => interaction.paperId === paperId && interaction.queued
      )
      .map((interaction) => usersMap.get(interaction.userId))
      .filter((user): user is User => user !== undefined);
  };

  // Get the number of users who have starred a specific paper
  const getStarCountForPaper = (paperId: string): number => {
    return interactions.filter(
      (interaction) => interaction.paperId === paperId && interaction.starred
    ).length;
  };

  // Handle mouse down on any column resize handle
  const handleMouseDown = useCallback(
    (e: React.MouseEvent, columnName: string) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
      isDraggingRef.current = true;
      draggedColumnRef.current = columnName;
      dragStartX.current = e.clientX;
      dragStartWidth.current =
        columnWidths[columnName as keyof typeof columnWidths];

      // Add global mouse event listeners
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    },
    [columnWidths]
  );

  // Handle mouse move during column resize
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDraggingRef.current || !draggedColumnRef.current) return;

    const deltaX = e.clientX - dragStartX.current;
    const newWidth = Math.max(100, dragStartWidth.current + deltaX);

    setColumnWidths((prev) => ({
      ...prev,
      [draggedColumnRef.current!]: newWidth,
    }));

    setMouseX(e.clientX);
  }, []);

  // Handle mouse up to end column resize
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    isDraggingRef.current = false;
    draggedColumnRef.current = null;

    // Remove global mouse event listeners
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", handleMouseUp);
  }, [handleMouseMove]);

  // Helper function to update URL with search parameters
  const updateUrlWithSearch = (searchValue: string) => {
    const params = new URLSearchParams();
    if (searchValue.trim()) {
      params.set("search", searchValue.trim());
    }
    const newUrl = params.toString()
      ? `/papers?${params.toString()}`
      : "/papers";
    router.push(newUrl, { scroll: false });
  };

  // Handle tag click to apply search filter
  const handleTagClick = (tag: string) => {
    if (!tag || typeof tag !== "string") {
      console.warn("Invalid tag provided to handleTagClick:", tag);
      return;
    }

    const trimmedTag = tag.trim();
    if (!trimmedTag) {
      console.warn("Empty tag provided to handleTagClick");
      return;
    }

    setSearchQuery(trimmedTag);
    updateUrlWithSearch(trimmedTag);
  };

  // Handle paper title click
  const handlePaperClick = (paper: PaperWithUserContext) => {
    if (!paper || typeof paper !== "object" || !paper.id) {
      console.warn("Invalid paper object provided to handlePaperClick:", paper);
      return;
    }

    openPaper(paper, users, interactions, handleToggleStar, handleToggleQueue);
  };

  // Handle user card close
  const handleUserCardClose = () => {
    setSelectedUser(null);
  };

  // Handle user avatar click
  const handleUserClick = (user: User) => {
    setSelectedUser(user);
  };

  // Handle toggle star
  const handleToggleStar = (paperId: string) => {
    starMutation.mutate(paperId);
  };

  // Handle toggle queue
  const handleToggleQueue = async (paperId: string) => {
    if (!paperId || typeof paperId !== "string") {
      console.warn("Invalid paperId provided to handleToggleQueue:", paperId);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("paperId", paperId);
      await toggleQueueAction(formData);

      // Local state update is now handled by the overlay stack system
    } catch (error) {
      console.error("Error toggling queue:", error);
    }
  };

  // Handle loading state for large datasets
  useEffect(() => {
    if (papers.length > 100) {
      setIsLoading(true);
      // Use setTimeout to allow UI to update
      const timeoutId = setTimeout(() => setIsLoading(false), 0);
      return () => clearTimeout(timeoutId); // Cleanup timeout
    } else {
      setIsLoading(false);
    }
  }, [papers.length]);

  // Filter and sort papers
  const filteredAndSortedPapers = useMemo(() => {
    let filtered = papers;

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((paper) => {
        // Search in title
        if (paper.title.toLowerCase().includes(query)) return true;

        // Search in tags
        if (
          paper.tags &&
          paper.tags.some((tag) => tag.toLowerCase().includes(query))
        )
          return true;

        // Search in user names and emails from "read by" column
        const readers = getReadersForPaper(paper.id);
        if (
          readers.some(
            (user) =>
              (user.name && user.name.toLowerCase().includes(query)) ||
              (user.email && user.email.toLowerCase().includes(query))
          )
        )
          return true;

        // Search in user names and emails from "queued" column
        const queuedUsers = getQueuedForPaper(paper.id);
        if (
          queuedUsers.some(
            (user) =>
              (user.name && user.name.toLowerCase().includes(query)) ||
              (user.email && user.email.toLowerCase().includes(query))
          )
        )
          return true;

        return false;
      });
    }

    // Filter by starred status
    if (showOnlyStarred) {
      filtered = filtered.filter((paper) => paper.isStarredByCurrentUser);
    }

    // Sort papers
    return filtered.sort((a, b) => {
      let aValue: string | number;
      let bValue: string | number;

      switch (sortColumn) {
        case "title":
          aValue = a.title.toLowerCase();
          bValue = b.title.toLowerCase();
          break;
        case "tags":
          // Sort by first tag
          aValue = a.tags && a.tags.length > 0 ? a.tags[0].toLowerCase() : "";
          bValue = b.tags && b.tags.length > 0 ? b.tags[0].toLowerCase() : "";
          break;
        case "readBy":
          // Sort by number of readers
          aValue = getReadersForPaper(a.id).length;
          bValue = getReadersForPaper(b.id).length;
          break;
        case "queued":
          // Sort by number of queued users
          aValue = getQueuedForPaper(a.id).length;
          bValue = getQueuedForPaper(b.id).length;
          break;

        default:
          aValue = a.title.toLowerCase();
          bValue = b.title.toLowerCase();
      }

      // Handle string comparison
      if (typeof aValue === "string" && typeof bValue === "string") {
        return sortDirection === "asc"
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      // Handle number comparison
      if (typeof aValue === "number" && typeof bValue === "number") {
        return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
      }

      return 0;
    });
  }, [papers, searchQuery, showOnlyStarred, sortColumn, sortDirection]);

  // Handle column sorting
  const handleSort = (col: string) => {
    if (sortColumn === col) {
      setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortColumn(col as "title" | "tags" | "readBy" | "queued");
      setSortDirection("asc");
    }
  };

  // Render sort indicator for column headers
  const renderSortIndicator = (columnName: string) => (
    <span className="ml-1">
      {sortColumn === columnName ? (
        sortDirection === "asc" ? (
          "↑"
        ) : (
          "↓"
        )
      ) : (
        <span className="text-gray-400">↕</span>
      )}
    </span>
  );

  // Render user avatar with highlighting
  const renderUserAvatar = (
    user: User,
    bgColor: string,
    textColor: string,
    index: number = 0
  ) => {
    const userName = user.name || "Unknown User";
    const initials = userName
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);

    // Check if user matches search query (only when there's a query)
    const query = searchQuery.toLowerCase().trim();
    const matchesQuery =
      query &&
      ((user.name && user.name.toLowerCase().includes(query)) ||
        (user.email && user.email.toLowerCase().includes(query)));

    return (
      <button
        key={user.id}
        onClick={() => handleUserClick(user)}
        className={`h-6 w-6 rounded-full ${matchesQuery ? "bg-yellow-200 text-black" : "bg-blue-600 text-white"} flex cursor-pointer items-center justify-center border border-white text-xs font-semibold transition-transform hover:scale-110`}
        title={`Click to view ${userName}'s profile`}
      >
        {user.image || initials}
      </button>
    );
  };

  // Highlight text that matches the search query
  const highlightText = (text: string, query: string): React.ReactNode => {
    if (!query.trim()) return text;

    try {
      const regex = new RegExp(`(${query})`, "gi");
      const parts = text.split(regex);

      return parts.map((part, index) =>
        regex.test(part) ? (
          <mark key={index} className="rounded bg-yellow-200">
            {part}
          </mark>
        ) : (
          part
        )
      );
    } catch (error) {
      console.warn("Error highlighting text:", error);
      return text;
    }
  };

  // Column configurations for the table
  const columnConfigs: ColumnConfig[] = useMemo(
    () => [
      {
        key: "title",
        label: "Title",
        width: columnWidths.title,
        minWidth: 200,
        maxWidth: 800,
        sortable: true,
        sticky: true, // Frozen column
        renderHeader: (sortIndicator) => (
          <div className="flex items-center justify-between">
            <span>Title</span>
            {sortIndicator}
          </div>
        ),
        renderCell: (paper) => {
          const starCount = getStarCountForPaper(paper.id);

          return (
            <div className="flex items-center gap-2">
              <StarWidgetQuery
                paperId={paper.id}
                initialTotalStars={starCount}
                initialIsStarredByCurrentUser={paper.isStarredByCurrentUser}
                size="md"
              />
              <button
                className="hover:text-primary-600 block truncate text-left transition-colors"
                onClick={() => handlePaperClick(paper)}
                title={paper.title}
              >
                {highlightText(paper.title, searchQuery)}
              </button>
            </div>
          );
        },
      },
      {
        key: "tags",
        label: "Tags",
        width: columnWidths.tags,
        minWidth: 100,
        maxWidth: 300,
        sortable: true,
        renderHeader: (sortIndicator) => (
          <div className="flex items-center justify-between">
            <span>Tags</span>
            {sortIndicator}
          </div>
        ),
        renderCell: (paper) => {
          const maxVisibleTags = 6; // Show up to 6 tags before truncating
          const tags = paper.tags || [];
          const visibleTags = tags.slice(0, maxVisibleTags);
          const hiddenTagsCount = tags.length - maxVisibleTags;

          return (
            <div className="flex max-h-16 flex-wrap gap-1 overflow-hidden">
              {visibleTags.map((tag, idx) => (
                <button
                  key={idx}
                  onClick={() => handleTagClick(tag)}
                  className="inline-block flex-shrink-0 cursor-pointer rounded px-2 py-0.5 text-xs font-bold transition-colors hover:bg-gray-200"
                  style={{ backgroundColor: "#EFF3F9", color: "#131720" }}
                  title={`Click to filter by "${tag}"`}
                >
                  {highlightText(tag, searchQuery)}
                </button>
              ))}
              {hiddenTagsCount > 0 && (
                <span className="flex-shrink-0 text-xs text-gray-500">
                  +{hiddenTagsCount} more
                </span>
              )}
            </div>
          );
        },
      },
      {
        key: "readBy",
        label: "Read by",
        width: columnWidths.readBy,
        minWidth: 100,
        maxWidth: 200,
        sortable: true,
        renderHeader: (sortIndicator) => (
          <div className="flex items-center justify-between">
            <span>Read by</span>
            {sortIndicator}
          </div>
        ),
        renderCell: (paper) => (
          <div className="flex gap-0.5">
            {getReadersForPaper(paper.id).length > 0
              ? getReadersForPaper(paper.id).map((user, index) =>
                  renderUserAvatar(user, "", "", index)
                )
              : null}
          </div>
        ),
      },

      {
        key: "queued",
        label: "Queued",
        width: columnWidths.queued,
        minWidth: 100,
        maxWidth: 200,
        sortable: true,
        renderHeader: (sortIndicator) => (
          <div className="flex items-center justify-between">
            <span>Queued</span>
            {sortIndicator}
          </div>
        ),
        renderCell: (paper) => (
          <div className="flex gap-0.5">
            {getQueuedForPaper(paper.id).length > 0
              ? getQueuedForPaper(paper.id).map((user, index) =>
                  renderUserAvatar(user, "", "", index)
                )
              : null}
          </div>
        ),
      },
    ],
    [
      columnWidths,
      sortColumn,
      sortDirection,
      searchQuery,
      handleTagClick,
      renderUserAvatar,
      highlightText,
      getStarCountForPaper,
    ]
  );

  return (
    <div className="flex h-full flex-col">
      {/* Header Section - matches NewPostForm styling */}
      <div className="border-b border-gray-200 bg-white p-4 md:p-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900">Papers</h1>
          <span className="text-sm text-gray-500">
            {filteredAndSortedPapers.length} papers
          </span>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-2 md:p-4">
        <div className="w-full max-w-full overflow-hidden px-1 md:px-2">
          {/* Filter and Sort Controls */}
          <div className="mb-6 max-w-full space-y-4">
            {/* Search Input */}
            <div className="relative max-w-full">
              {/* Search Icon - positioned absolutely in the left side of the input */}
              <svg
                className="absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2 transform text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <circle cx="11" cy="11" r="8" />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="m21 21-4.35-4.35"
                />
              </svg>

              {/* Search Input Field */}
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  const value = e.target.value;
                  // Validate input before calling callback
                  if (typeof value === "string") {
                    setSearchQuery(value);
                  }
                }}
                placeholder="Filter..."
                className="focus:ring-primary-500 w-full rounded-lg border border-gray-300 py-2 pr-10 pl-10 focus:border-transparent focus:ring-2"
                maxLength={200} // Prevent extremely long search queries
              />

              {/* Clear Button - only show when there's content */}
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute top-1/2 right-3 h-5 w-5 -translate-y-1/2 transform text-gray-400 transition-colors hover:text-gray-600"
                  aria-label="Clear search"
                  type="button"
                >
                  <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              )}
            </div>

            {/* Starred Filter */}
            <div className="flex flex-wrap items-center gap-4">
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={showOnlyStarred}
                  onChange={(e) => setShowOnlyStarred(e.target.checked)}
                  className="text-primary-600 focus:ring-primary-500 rounded border-gray-300"
                />
                <span className="text-xs text-gray-700 md:text-sm">
                  Show only starred (
                  {
                    papers.filter((paper) => paper.isStarredByCurrentUser)
                      .length
                  }
                  )
                </span>
              </label>
            </div>
          </div>

          {/* Mobile Card View - Hidden on MD+ screens */}
          <div className="space-y-3 md:hidden">
            {filteredAndSortedPapers.map((paper) => {
              if (!paper || typeof paper !== "object" || !paper.id) {
                return null;
              }

              const starCount = getStarCountForPaper(paper.id);
              const readers = getReadersForPaper(paper.id);
              const queuedUsers = getQueuedForPaper(paper.id);

              return (
                <div
                  key={paper.id}
                  className="rounded-lg border border-gray-200 bg-white p-3"
                >
                  {/* Header with title and star */}
                  <div className="mb-2 flex items-start gap-2">
                    <StarWidgetQuery
                      paperId={paper.id}
                      initialTotalStars={starCount}
                      initialIsStarredByCurrentUser={
                        paper.isStarredByCurrentUser
                      }
                      size="sm"
                    />
                    <button
                      className="hover:text-primary-600 flex-1 text-left text-sm leading-tight font-medium transition-colors"
                      onClick={() => handlePaperClick(paper)}
                    >
                      {highlightText(paper.title, searchQuery)}
                    </button>
                  </div>

                  {/* Tags */}
                  {paper.tags && paper.tags.length > 0 && (
                    <div className="mb-2 flex flex-wrap gap-1">
                      {paper.tags.slice(0, 4).map((tag, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleTagClick(tag)}
                          className="inline-block cursor-pointer rounded px-2 py-0.5 text-xs font-bold transition-colors hover:bg-gray-200"
                          style={{
                            backgroundColor: "#EFF3F9",
                            color: "#131720",
                          }}
                        >
                          {highlightText(tag, searchQuery)}
                        </button>
                      ))}
                      {paper.tags.length > 4 && (
                        <span className="text-xs text-gray-500">
                          +{paper.tags.length - 4} more
                        </span>
                      )}
                    </div>
                  )}

                  {/* Users - Readers and Queued */}
                  <div className="flex gap-4 text-xs text-gray-600">
                    {readers.length > 0 && (
                      <div className="flex items-center gap-1">
                        <span>Read by:</span>
                        <div className="flex gap-0.5">
                          {readers
                            .slice(0, 3)
                            .map((user, index) =>
                              renderUserAvatar(user, "", "", index)
                            )}
                          {readers.length > 3 && (
                            <span className="text-gray-500">
                              +{readers.length - 3}
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                    {queuedUsers.length > 0 && (
                      <div className="flex items-center gap-1">
                        <span>Queued:</span>
                        <div className="flex gap-0.5">
                          {queuedUsers
                            .slice(0, 3)
                            .map((user, index) =>
                              renderUserAvatar(user, "", "", index)
                            )}
                          {queuedUsers.length > 3 && (
                            <span className="text-gray-500">
                              +{queuedUsers.length - 3}
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {filteredAndSortedPapers.length === 0 && (
              <div className="py-8 text-center text-gray-500">
                No papers found matching your criteria.
              </div>
            )}
          </div>

          {/* Papers Table Container - Hidden on mobile */}
          <div className="relative hidden md:block">
            {/* Loading indicator */}
            {isLoading && (
              <div className="bg-opacity-75 absolute inset-0 z-20 flex items-center justify-center bg-white">
                <div className="text-gray-600">Filtering papers...</div>
              </div>
            )}

            {/* Width indicator during drag */}
            {isDragging && draggedColumnRef.current && (
              <div
                className="pointer-events-none fixed z-50 rounded bg-blue-500 px-2 py-1 text-xs text-white"
                style={{
                  left: mouseX,
                  top: 10,
                }}
              >
                {
                  columnWidths[
                    draggedColumnRef.current as keyof typeof columnWidths
                  ]
                }
                px
              </div>
            )}

            {/* User Card */}
            {selectedUser && (
              <UserCard
                user={selectedUser}
                allPapers={papers}
                users={users}
                interactions={interactions}
                onClose={handleUserCardClose}
              />
            )}

            {/* Scroll Container with Table */}
            <div className="overflow-auto rounded-lg border border-gray-200">
              <table
                ref={tableRef}
                className="w-full bg-white text-sm"
                style={{ tableLayout: "fixed" }}
              >
                {/* Column Group for Width Management */}
                <colgroup>
                  {columnConfigs.map((config) => (
                    <col
                      key={config.key}
                      style={{ width: `${config.width}px` }}
                    />
                  ))}
                </colgroup>

                {/* Table Header */}
                <thead>
                  <tr className="bg-gray-50">
                    {columnConfigs.map((config) => (
                      <th
                        key={config.key}
                        className={`relative px-4 py-2 text-left ${
                          config.sticky ? "sticky left-0 z-10 bg-gray-50" : ""
                        } ${
                          config.sortable
                            ? "cursor-pointer transition-colors hover:bg-gray-100"
                            : ""
                        }`}
                        style={{
                          position: config.sticky ? "sticky" : "static",
                          left: config.sticky ? 0 : "auto",
                          top: 0,
                          zIndex: config.sticky ? 10 : "auto",
                        }}
                        onClick={() => {
                          if (config.sortable && !isDragging) {
                            handleSort(config.key);
                          }
                        }}
                      >
                        <div className="flex items-center justify-between">
                          {config.renderHeader(renderSortIndicator(config.key))}
                        </div>
                        {/* Resize handle */}
                        <div
                          className="absolute top-0 right-0 bottom-0 z-20 w-2 cursor-col-resize transition-colors hover:bg-blue-400"
                          onMouseDown={(e) => handleMouseDown(e, config.key)}
                          title="Drag to resize column"
                          style={{
                            cursor: "col-resize",
                            userSelect: "none",
                          }}
                        />
                      </th>
                    ))}
                  </tr>
                </thead>

                {/* Table Body */}
                <tbody>
                  {filteredAndSortedPapers.map((paper) => {
                    // Validate paper object before rendering
                    if (!paper || typeof paper !== "object" || !paper.id) {
                      console.warn("Invalid paper object found:", paper);
                      return null;
                    }

                    return (
                      <tr
                        key={paper.id}
                        className="border-t border-gray-100 hover:bg-gray-50"
                      >
                        {columnConfigs.map((config) => (
                          <td
                            key={config.key}
                            className={`px-4 py-2 ${
                              config.sticky ? "sticky left-0 z-10 bg-white" : ""
                            }`}
                            style={{
                              position: config.sticky ? "sticky" : "static",
                              left: config.sticky ? 0 : "auto",
                              zIndex: config.sticky ? 10 : "auto",
                            }}
                          >
                            {config.renderCell(paper)}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
