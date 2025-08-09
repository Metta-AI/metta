"use client";

import React, { FC, useState, useEffect, useRef, useMemo } from "react";
import { InstitutionDTO } from "@/posts/data/institutions-client";
import { useOverlayNavigation } from "./OverlayStack";

interface InstitutionsViewProps {
  institutions: InstitutionDTO[];
}

/**
 * InstitutionsView Component
 *
 * Displays a grid of institution cards with filtering and sorting capabilities.
 * Shows institution statistics, recent papers, and top authors.
 */
export const InstitutionsView: FC<InstitutionsViewProps> = ({
  institutions,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState("paperCount");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");
  const [institutionsList, setInstitutionsList] =
    useState<InstitutionDTO[]>(institutions);
  const filterInputRef = useRef<HTMLInputElement>(null);
  const { openInstitution } = useOverlayNavigation();

  // Update institutions list when props change
  useEffect(() => {
    setInstitutionsList(institutions);
  }, [institutions]);

  // Filter and sort institutions based on current state
  const filteredAndSortedInstitutions = useMemo(() => {
    let filtered = institutionsList;

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (institution) =>
          institution.name.toLowerCase().includes(query) ||
          institution.topCategories.some((cat) =>
            cat.toLowerCase().includes(query)
          ) ||
          institution.authors.some((author) =>
            author.name.toLowerCase().includes(query)
          )
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue: string | number;
      let bValue: string | number;

      switch (sortBy) {
        case "name":
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
          break;
        case "paperCount":
          aValue = a.paperCount;
          bValue = b.paperCount;
          break;
        case "authorCount":
          aValue = a.authorCount;
          bValue = b.authorCount;
          break;
        case "avgStars":
          aValue = a.avgStars;
          bValue = b.avgStars;
          break;
        case "recentActivity":
          aValue = a.recentActivity ? new Date(a.recentActivity).getTime() : 0;
          bValue = b.recentActivity ? new Date(b.recentActivity).getTime() : 0;
          break;
        default:
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
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

    return filtered;
  }, [institutionsList, searchQuery, sortBy, sortDirection]);

  const getInstitutionInitials = (name: string) => {
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 3);
  };

  const handleInstitutionClick = (institution: InstitutionDTO) => {
    // Convert recent papers to PaperWithUserContext format (simplified)
    const papers = institution.recentPapers.map((paper) => ({
      ...paper,
      abstract: null,
      institutions: [institution.name],
      tags: [],
      source: null,
      externalId: null,
      isStarredByCurrentUser: false,
      isQueuedByCurrentUser: false,
      updatedAt: new Date(),
    }));

    // Convert authors to AuthorDTO format (simplified)
    const authors = institution.authors.map((author) => ({
      id: author.id,
      name: author.name,
      username: null,
      email: null,
      avatar: null,
      institution: institution.name,
      department: null,
      title: null,
      expertise: [],
      hIndex: null,
      totalCitations: null,
      claimed: false,
      isFollowing: false,
      recentActivity: null,
      orcid: null,
      googleScholarId: null,
      arxivId: null,
      createdAt: new Date(),
      updatedAt: new Date(),
      paperCount: author.paperCount,
      recentPapers: [],
    }));

    openInstitution(institution.name, papers, authors);
  };

  const handleSortClick = (key: string) => {
    if (sortBy === key) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortBy(key);
      setSortDirection(key === "name" ? "asc" : "desc");
    }
  };

  const formatDate = (date: Date | string | null) => {
    if (!date) return "Unknown";
    const dateObj = typeof date === "string" ? new Date(date) : date;
    if (isNaN(dateObj.getTime())) return "Unknown";

    const now = new Date();
    const diffInDays = Math.floor(
      (now.getTime() - dateObj.getTime()) / (1000 * 60 * 60 * 24)
    );

    if (diffInDays === 0) return "Today";
    if (diffInDays === 1) return "Yesterday";
    if (diffInDays < 7) return `${diffInDays} days ago`;
    if (diffInDays < 30) return `${Math.floor(diffInDays / 7)} weeks ago`;
    return `${Math.floor(diffInDays / 30)} months ago`;
  };

  const sortOptions = [
    { key: "paperCount", label: "Papers" },
    { key: "authorCount", label: "Authors" },
    { key: "avgStars", label: "Avg Stars" },
    { key: "name", label: "Name" },
    { key: "recentActivity", label: "Recent Activity" },
  ];

  return (
    <div className="p-4">
      {/* Filter and Sort Controls */}
      <div className="mb-6 space-y-4">
        {/* Search Input */}
        <div className="relative">
          <svg
            className="absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2 transform text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <circle cx="11" cy="11" r="8" />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="m21 21-4.35-4.35"
            />
          </svg>

          <input
            ref={filterInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search institutions..."
            className="focus:ring-primary-500 w-full rounded-lg border border-gray-300 py-2 pr-10 pl-10 focus:border-transparent focus:ring-2"
          />

          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute top-1/2 right-3 h-5 w-5 -translate-y-1/2 transform text-gray-400 transition-colors hover:text-gray-600"
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

        {/* Sort Controls */}
        <div className="flex flex-wrap items-center gap-4">
          <span className="text-sm font-medium text-gray-700">Sort by:</span>
          {sortOptions.map((option) => (
            <button
              key={option.key}
              onClick={() => handleSortClick(option.key)}
              className={`rounded-full px-3 py-1 text-sm transition-colors ${
                sortBy === option.key
                  ? "bg-primary-100 text-primary-800"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              {option.label}
              {sortBy === option.key && (
                <span className="ml-1">
                  {sortDirection === "asc" ? "↑" : "↓"}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Results Count */}
        <div className="text-sm text-gray-600">
          {filteredAndSortedInstitutions.length} institution
          {filteredAndSortedInstitutions.length !== 1 ? "s" : ""}
        </div>
      </div>

      {/* Institutions Grid */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredAndSortedInstitutions.map((institution) => (
          <div
            key={institution.name}
            className="cursor-pointer rounded-lg border border-gray-200 bg-white p-6 transition-all hover:border-gray-300 hover:shadow-md"
            onClick={() => handleInstitutionClick(institution)}
          >
            {/* Institution Header */}
            <div className="mb-4 flex items-start gap-4">
              <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-blue-600 text-sm font-semibold text-white">
                {getInstitutionInitials(institution.name)}
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="truncate text-lg font-semibold text-gray-900">
                  {institution.name}
                </h3>
                <p className="text-sm text-gray-600">
                  {institution.authorCount} authors • {institution.paperCount}{" "}
                  papers
                </p>
              </div>
            </div>

            {/* Statistics */}
            <div className="mb-4 grid grid-cols-2 gap-4">
              <div className="rounded-lg bg-gray-50 p-3 text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {institution.paperCount}
                </div>
                <div className="text-xs text-gray-600">Papers</div>
              </div>
              <div className="rounded-lg bg-gray-50 p-3 text-center">
                <div className="text-2xl font-bold text-green-600">
                  {institution.avgStars}
                </div>
                <div className="text-xs text-gray-600">Avg Stars</div>
              </div>
            </div>

            {/* Top Categories */}
            {institution.topCategories.length > 0 && (
              <div className="mb-4">
                <h4 className="mb-2 text-sm font-medium text-gray-700">
                  Top Areas
                </h4>
                <div className="flex flex-wrap gap-1">
                  {institution.topCategories
                    .slice(0, 3)
                    .map((category, idx) => (
                      <span
                        key={idx}
                        className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-700"
                      >
                        {category}
                      </span>
                    ))}
                  {institution.topCategories.length > 3 && (
                    <span className="text-xs text-gray-500">
                      +{institution.topCategories.length - 3} more
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Recent Activity */}
            <div className="text-xs text-gray-500">
              Last activity: {formatDate(institution.recentActivity)}
            </div>
          </div>
        ))}
      </div>

      {/* Empty State */}
      {filteredAndSortedInstitutions.length === 0 && (
        <div className="py-8 text-center">
          <svg
            className="mx-auto mb-4 h-12 w-12 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0h3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"
            />
          </svg>
          <p className="text-gray-500">
            {searchQuery
              ? "No institutions found matching your search."
              : "No institutions found."}
          </p>
        </div>
      )}
    </div>
  );
};
