"use client";

import React, { FC, useState, useEffect, useRef, useMemo } from "react";
import {
  Plus,
  Search,
  ChevronDown,
  Users,
  Star,
  Calendar,
  ExternalLink,
} from "lucide-react";

import { InstitutionCreateForm } from "./InstitutionCreateForm";
import { InstitutionManagementModal } from "./InstitutionManagementModal";
import { UnifiedInstitutionDTO } from "@/posts/data/managed-institutions";
import { useOverlayNavigation } from "./OverlayStack";

interface UnifiedInstitutionsViewProps {
  userInstitutions: UnifiedInstitutionDTO[];
  allInstitutions: UnifiedInstitutionDTO[];
}

export const UnifiedInstitutionsView: FC<UnifiedInstitutionsViewProps> = ({
  userInstitutions,
  allInstitutions,
}) => {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedInstitution, setSelectedInstitution] =
    useState<UnifiedInstitutionDTO | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState("paperCount");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");
  const filterInputRef = useRef<HTMLInputElement>(null);
  const { openInstitution } = useOverlayNavigation();

  // Combine all institutions for the main view
  const allInstitutionsForView = allInstitutions;

  // Helper to get institution with full member data
  const getInstitutionWithMembers = (institution: UnifiedInstitutionDTO) => {
    // For user institutions (where user is a member), get the version with full member data
    const userInstitution = userInstitutions.find(
      (ui) => ui.id === institution.id
    );
    return userInstitution || institution;
  };

  // Filter and sort institutions based on current state
  const filteredAndSortedInstitutions = useMemo(() => {
    let filtered = allInstitutionsForView;

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (institution) =>
          institution.name.toLowerCase().includes(query) ||
          institution.topCategories.some((cat) =>
            cat.toLowerCase().includes(query)
          ) ||
          institution.authors?.some((author) =>
            author.name.toLowerCase().includes(query)
          )
      );
    }

    // Apply sorting
    const sortedFiltered = [...filtered].sort((a, b) => {
      let valueA: any, valueB: any;

      switch (sortBy) {
        case "name":
          valueA = a.name.toLowerCase();
          valueB = b.name.toLowerCase();
          break;
        case "paperCount":
          valueA = a.paperCount;
          valueB = b.paperCount;
          break;
        case "authorCount":
          valueA = a.authorCount;
          valueB = b.authorCount;
          break;
        case "avgStars":
          valueA = a.avgStars;
          valueB = b.avgStars;
          break;
        case "recentActivity":
          valueA = a.recentActivity ? new Date(a.recentActivity).getTime() : 0;
          valueB = b.recentActivity ? new Date(b.recentActivity).getTime() : 0;
          break;
        default:
          valueA = a.paperCount;
          valueB = b.paperCount;
      }

      if (sortDirection === "asc") {
        return valueA > valueB ? 1 : valueA < valueB ? -1 : 0;
      } else {
        return valueA < valueB ? 1 : valueA > valueB ? -1 : 0;
      }
    });

    return sortedFiltered;
  }, [allInstitutionsForView, searchQuery, sortBy, sortDirection]);

  const getInstitutionInitials = (name: string) => {
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 3);
  };

  const handleInstitutionClick = (institution: UnifiedInstitutionDTO) => {
    if (institution.recentPapers && institution.authors) {
      // Convert recent papers to expected format
      const papers = institution.recentPapers.map((paper) => ({
        ...paper,
        abstract: null,
        institutions: [institution.name],
        tags: [],
        source: null,
        externalId: null,
        starred: null,
        isStarredByCurrentUser: false,
        isQueuedByCurrentUser: false,
        createdAt: new Date(paper.createdAt),
        updatedAt: new Date(),
      }));

      // Convert authors to expected format
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
    }
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
      {/* Header with Create Button */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Institutions</h1>
          <p className="text-gray-600">
            Research institutions, companies, and organizations
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700"
        >
          <Plus className="h-4 w-4" />
          Create Institution
        </button>
      </div>

      {/* Filter and Sort Controls */}
      <div className="mb-6 space-y-4">
        {/* Search Input */}
        <div className="relative">
          <Search className="absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2 transform text-gray-400" />
          <input
            ref={filterInputRef}
            type="text"
            placeholder="Search institutions, categories, or authors..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-lg border border-gray-300 py-3 pr-4 pl-10 text-gray-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
          />
        </div>

        {/* Sort Options */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-gray-700">Sort by:</span>
          {sortOptions.map((option) => (
            <button
              key={option.key}
              onClick={() => handleSortClick(option.key)}
              className={`flex items-center gap-1 rounded-full px-3 py-1 text-sm transition-colors ${
                sortBy === option.key
                  ? "bg-blue-100 text-blue-700"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              {option.label}
              {sortBy === option.key && (
                <ChevronDown
                  className={`h-4 w-4 transition-transform ${
                    sortDirection === "asc" ? "rotate-180" : ""
                  }`}
                />
              )}
            </button>
          ))}
        </div>

        {/* Results Count */}
        <div className="text-sm text-gray-500">
          Showing {filteredAndSortedInstitutions.length} of{" "}
          {allInstitutionsForView.length} institutions
        </div>
      </div>

      {/* Institutions Grid - Original Styling */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredAndSortedInstitutions.map((institution) => (
          <div
            key={institution.id}
            className="cursor-pointer rounded-lg border border-gray-200 bg-white p-6 transition-all hover:border-gray-300 hover:shadow-md"
            onClick={() => handleInstitutionClick(institution)}
          >
            {/* Institution Header */}
            <div className="mb-4 flex items-start gap-4">
              <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-blue-600 text-sm font-semibold text-white">
                {getInstitutionInitials(institution.name)}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between">
                  <h3 className="truncate text-lg font-semibold text-gray-900">
                    {institution.name}
                  </h3>
                  {institution.currentUserRole === "admin" && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedInstitution(
                          getInstitutionWithMembers(institution)
                        );
                      }}
                      className="ml-2 rounded-md p-1.5 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
                      title="Manage institution"
                    >
                      <Users className="h-4 w-4" />
                    </button>
                  )}
                </div>
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
                  {institution.avgStars.toFixed(1)}
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

            {/* User Info (if applicable) */}
            {institution.currentUserRole && (
              <div className="mb-4">
                <div className="flex items-center gap-2 text-xs">
                  <span className="rounded bg-blue-100 px-2 py-1 font-medium text-blue-700">
                    {institution.currentUserRole}
                  </span>
                  {institution.memberCount > 0 && (
                    <span className="text-gray-500">
                      {institution.memberCount} member
                      {institution.memberCount !== 1 ? "s" : ""}
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

      {/* Modals */}
      <InstitutionCreateForm
        isOpen={showCreateForm}
        onClose={() => setShowCreateForm(false)}
      />

      {selectedInstitution && (
        <InstitutionManagementModal
          isOpen={!!selectedInstitution}
          onClose={() => setSelectedInstitution(null)}
          institution={selectedInstitution}
          currentUserRole={selectedInstitution.currentUserRole}
        />
      )}
    </div>
  );
};
