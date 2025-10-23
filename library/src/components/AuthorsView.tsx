"use client";

import React, { FC, useState, useEffect, useRef, useMemo } from "react";
import { Search } from "lucide-react";

import { AuthorDTO } from "@/posts/data/authors-client";
import { useOverlayNavigation } from "./OverlayStack";
import { AuthorCard } from "@/components/authors/AuthorCard";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useFilterSort } from "@/lib/hooks/useFilterSort";
import { SortControls } from "@/components/ui/sort-controls";

interface AuthorsViewProps {
  authors: AuthorDTO[];
}

/**
 * AuthorsView Component
 *
 * Displays a grid of author cards with filtering and sorting capabilities.
 * Matches the mockup design exactly with filter controls and no header.
 */
export const AuthorsView: FC<AuthorsViewProps> = ({ authors }) => {
  const [authorsList, setAuthorsList] = useState<AuthorDTO[]>(authors);
  const filterInputRef = useRef<HTMLInputElement>(null);
  const { openAuthor } = useOverlayNavigation();

  // Update authors list when props change
  useEffect(() => {
    setAuthorsList(authors);
  }, [authors]);

  const {
    searchQuery,
    setSearchQuery,
    sortBy,
    setSortBy,
    sortDirection,
    setSortDirection,
    filteredAndSortedItems: filteredAndSortedAuthors,
  } = useFilterSort<AuthorDTO>(authorsList, {
    getSearchableValues: (author) => [
      author.name,
      author.institution ?? "",
      ...author.expertise,
    ],
    sorters: {
      name: (author) => author.name.toLowerCase(),
      institution: (author) => author.institution?.toLowerCase() || "",
      papers: (author) => author.paperCount || 0,
      hIndex: (author) => author.hIndex || 0,
      citations: (author) => author.totalCitations || 0,
    },
    initialSortKey: "name",
    initialSortDirection: "asc",
  });

  const handleAuthorClick = (author: AuthorDTO) => {
    openAuthor(author);
  };

  const handleToggleFollow = (author: AuthorDTO) => {
    setAuthorsList((prevAuthors) =>
      prevAuthors.map((a) =>
        a.id === author.id ? { ...a, isFollowing: !a.isFollowing } : a
      )
    );
  };

  const handleExpertiseClick = (expertise: string) => {
    setSearchQuery(expertise);
    filterInputRef.current?.focus();
  };

  const sortOptions = useMemo(
    () => [
      { key: "name", label: "Name" },
      { key: "institution", label: "Institution" },
      { key: "papers", label: "Papers" },
      { key: "hIndex", label: "h-index" },
      { key: "citations", label: "Citations" },
    ],
    []
  );

  return (
    <div className="flex h-full w-full flex-col">
      {/* Header Section */}
      <div className="border-b border-gray-200 bg-white px-4 py-4 md:px-6 md:py-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900">Authors</h1>
          <span className="text-sm text-gray-500">
            Showing {filteredAndSortedAuthors.length} of {authorsList.length}{" "}
            authors
          </span>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-6">
        <div className="mx-auto w-full max-w-7xl space-y-6">
          {/* Stats Bar */}
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary">
              {authorsList.filter((a) => a.claimed).length} claimed profiles
            </Badge>
            <Badge variant="secondary">
              {authorsList.filter((a) => a.isFollowing).length} following
            </Badge>
          </div>

          {/* Search Bar */}
          <div className="relative w-full">
            <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2" />
            <Input
              ref={filterInputRef}
              placeholder="Search authors"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className="pl-10"
            />
          </div>

          {/* Sort Controls */}
          <SortControls
            sortOptions={sortOptions}
            sortBy={sortBy}
            sortDirection={sortDirection}
            onSortChange={(key) => {
              setSortBy(key);
              setSortDirection(key === "name" ? "asc" : "desc");
            }}
            onDirectionToggle={() =>
              setSortDirection(sortDirection === "asc" ? "desc" : "asc")
            }
          />

          {/* Authors Grid */}
          {filteredAndSortedAuthors.length > 0 ? (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredAndSortedAuthors.map((author) => (
                <AuthorCard
                  key={author.id}
                  author={author}
                  onClick={handleAuthorClick}
                  onToggleFollow={handleToggleFollow}
                  onExpertiseClick={handleExpertiseClick}
                />
              ))}
            </div>
          ) : (
            <div className="border-border bg-card text-muted-foreground rounded-xl border p-8 text-center">
              No authors found.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
