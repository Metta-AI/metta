"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { useStarMutation } from "@/hooks/useStarMutation";
import { PapersTable } from "@/components/PapersTable";
import type { PaperSummary } from "@/lib/api/resources/papers";
import { useOverlayNavigation } from "@/components/OverlayStack";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";

import type {
  PaperWithUserContext,
  User,
  UserInteraction,
} from "@/posts/data/papers";

interface PapersViewProps {
  papers: PaperWithUserContext[];
  users: User[];
  interactions: UserInteraction[];
  initialSearch?: string;
}

export function PapersView({
  papers,
  users,
  interactions,
  initialSearch = "",
}: PapersViewProps) {
  const router = useRouter();
  const { openPaper } = useOverlayNavigation();
  const starMutation = useStarMutation();

  const [searchQuery, setSearchQuery] = useState(initialSearch);
  const [showOnlyStarred, setShowOnlyStarred] = useState(false);

  useEffect(() => {
    setSearchQuery(initialSearch);
  }, [initialSearch]);

  const updateUrlWithSearch = useCallback(
    (value: string) => {
      const params = new URLSearchParams();
      if (value.trim()) {
        params.set("search", value.trim());
      }
      const next = params.toString()
        ? `/papers?${params.toString()}`
        : "/papers";
      router.push(next, { scroll: false });
    },
    [router]
  );

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      updateUrlWithSearch(searchQuery);
    }, 250);

    return () => window.clearTimeout(timeoutId);
  }, [searchQuery, updateUrlWithSearch]);

  const papersById = useMemo(
    () => new Map(papers.map((paper) => [paper.id, paper])),
    [papers]
  );

  const filteredPapers = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    return papers.filter((paper) => {
      if (showOnlyStarred && !paper.isStarredByCurrentUser) {
        return false;
      }

      if (query.length === 0) {
        return true;
      }

      const haystacks: string[] = [paper.title];

      if (paper.tags) {
        haystacks.push(...paper.tags);
      }

      if (paper.authors) {
        haystacks.push(...paper.authors.map((author) => author.name));
      }

      if (paper.institutions) {
        haystacks.push(...paper.institutions);
      }

      return haystacks.some((value) => value?.toLowerCase().includes(query));
    });
  }, [papers, searchQuery, showOnlyStarred]);

  const tableRows = useMemo<PaperSummary[]>(
    () =>
      filteredPapers.map((paper) => ({
        id: paper.id,
        title: paper.title,
        createdAt:
          paper.createdAt instanceof Date
            ? paper.createdAt.toISOString()
            : paper.createdAt,
        stars: paper.stars ?? 0,
        citationCount: 0,
        abstractSummary: paper.abstract ?? null,
        authors:
          paper.authors?.map((author) => ({
            id: author.id,
            name: author.name,
          })) ?? [],
        tags: paper.tags ?? [],
      })),
    [filteredPapers]
  );

  const handleToggleStar = useCallback(
    (paperId: string) => {
      starMutation.mutate({ paperId });
    },
    [starMutation]
  );

  const handleRowClick = useCallback(
    (summary: PaperSummary) => {
      const fullPaper = papersById.get(summary.id);
      if (!fullPaper) {
        return;
      }

      openPaper(fullPaper, users, interactions, handleToggleStar);
    },
    [handleToggleStar, interactions, openPaper, papersById, users]
  );

  const handleTagClick = useCallback((tag: string) => {
    setSearchQuery(tag);
  }, []);

  const handleClearFilters = useCallback(() => {
    setSearchQuery("");
    setShowOnlyStarred(false);
  }, []);

  const filteredCount = filteredPapers.length;
  const totalCount = papers.length;
  const hasFilters = searchQuery.trim().length > 0 || showOnlyStarred;

  return (
    <div className="flex h-full w-full flex-col">
      {/* Header Section */}
      <div className="border-b border-gray-200 bg-white px-4 py-4 md:px-6 md:py-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900">Papers</h1>
          <span className="text-sm text-gray-500">
            Showing {filteredCount} of {totalCount} papers
          </span>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-6">
        <div className="mx-auto w-full max-w-7xl space-y-6">
          {/* Search Bar */}
          <div className="w-full">
            <Label htmlFor="papers-search">Search</Label>
            <Input
              id="papers-search"
              placeholder="Search title, tags, or authors"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className="mt-1"
            />
          </div>

          {/* Filter Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Checkbox
                id="papers-starred-only"
                checked={showOnlyStarred}
                onCheckedChange={(checked) =>
                  setShowOnlyStarred(Boolean(checked))
                }
              />
              <Label htmlFor="papers-starred-only" className="text-sm">
                Show starred only
              </Label>
            </div>
            {hasFilters && (
              <Button variant="outline" size="sm" onClick={handleClearFilters}>
                Clear filters
              </Button>
            )}
          </div>

          <PapersTable
            papers={tableRows}
            onRowClick={handleRowClick}
            onTagClick={handleTagClick}
          />
        </div>
      </div>
    </div>
  );
}
