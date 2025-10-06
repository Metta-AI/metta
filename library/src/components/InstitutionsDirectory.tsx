"use client";

import React, { useCallback, useMemo, useState } from "react";
import { useAction } from "next-safe-action/hooks";
import { Search } from "lucide-react";

import { InstitutionCard } from "@/components/institutions/InstitutionCard";
import { SectionHeading } from "@/components/ui/section-heading";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useFilterSort } from "@/lib/hooks/useFilterSort";
import { useErrorHandling } from "@/lib/hooks/useErrorHandling";
import { useOverlayNavigation } from "@/components/OverlayStack";
import { InstitutionCreateForm } from "@/components/InstitutionCreateForm";
import { InstitutionManagementModal } from "@/components/InstitutionManagementModal";
import type { UnifiedInstitutionDTO } from "@/posts/data/managed-institutions";
import { joinInstitutionAction } from "@/institutions/actions/joinInstitutionAction";

interface InstitutionsDirectoryProps {
  directory: UnifiedInstitutionDTO[];
  memberships: UnifiedInstitutionDTO[];
}

type Mode = "directory" | "member" | "admin";

export const InstitutionsDirectory: React.FC<InstitutionsDirectoryProps> = ({
  directory,
  memberships,
}) => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedInstitution, setSelectedInstitution] =
    useState<UnifiedInstitutionDTO | null>(null);
  const { openInstitution } = useOverlayNavigation();

  const {
    error: joinError,
    setError: setJoinError,
    clearError: clearJoinError,
  } = useErrorHandling({
    fallbackMessage: "Failed to join institution.",
  });

  const { execute: joinInstitution, isExecuting: isJoining } = useAction(
    joinInstitutionAction,
    {
      onSuccess: () => {
        clearJoinError();
      },
      onError: (error) => setJoinError(error),
    }
  );

  const allInstitutions = useMemo<UnifiedInstitutionDTO[]>(() => {
    const byId = new Map<string, UnifiedInstitutionDTO>();

    directory.forEach((institution) => {
      byId.set(institution.id, {
        ...institution,
        members: undefined,
      });
    });

    memberships.forEach((membership) => {
      byId.set(membership.id, membership);
    });

    return Array.from(byId.values());
  }, [directory, memberships]);

  const {
    searchQuery,
    setSearchQuery,
    sortBy,
    setSortBy,
    sortDirection,
    setSortDirection,
    filteredAndSortedItems,
  } = useFilterSort<UnifiedInstitutionDTO>(allInstitutions, {
    getSearchableValues: (institution) => [
      institution.name,
      institution.domain ?? "",
      ...(institution.topCategories ?? []),
      ...(institution.authors?.map((author) => author.name) ?? []),
    ],
    sorters: {
      name: (institution) => institution.name.toLowerCase(),
      paperCount: (institution) => institution.paperCount,
      authorCount: (institution) => institution.authorCount,
      avgStars: (institution) => institution.avgStars,
      recentActivity: (institution) =>
        institution.recentActivity
          ? new Date(institution.recentActivity).getTime()
          : 0,
    },
    initialSortKey: "paperCount",
    initialSortDirection: "desc",
  });

  const handleJoin = useCallback(
    (institution: UnifiedInstitutionDTO) => {
      const formData = new FormData();
      formData.append("institutionId", institution.id);
      joinInstitution(formData);
    },
    [joinInstitution]
  );

  const handleOpenInstitution = useCallback(
    (institution: UnifiedInstitutionDTO) => {
      if (!institution.recentPapers || !institution.authors) {
        return;
      }

      openInstitution(
        institution.name,
        institution.recentPapers.map((paper) => ({
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
        })),
        institution.authors.map((author) => ({
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
        }))
      );
    },
    [openInstitution]
  );

  const sortOptions: Array<{ key: string; label: string }> = useMemo(
    () => [
      { key: "paperCount", label: "Papers" },
      { key: "authorCount", label: "Authors" },
      { key: "avgStars", label: "Avg Stars" },
      { key: "name", label: "Name" },
      { key: "recentActivity", label: "Recent Activity" },
    ],
    []
  );

  const membershipsSet = useMemo(
    () => new Set(memberships.map((institution) => institution.id)),
    [memberships]
  );

  return (
    <div className="flex h-full w-full flex-col gap-4">
      <header className="space-y-4">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-foreground text-2xl font-semibold">
              Institutions
            </h1>
            <p className="text-muted-foreground text-sm">
              Discover research organizations and manage your memberships
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary">{memberships.length} joined</Badge>
            <Badge variant="secondary">{directory.length} directory</Badge>
            <Button onClick={() => setShowCreateModal(true)} size="sm">
              New institution
            </Button>
          </div>
        </div>

        <SectionHeading
          title="Directory"
          description="Search, filter, and join institutions"
        />

        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="relative w-full md:max-w-sm">
            <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 h-5 w-5 -translate-y-1/2" />
            <Input
              placeholder="Search institutions"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className="pl-10"
            />
          </div>
          <div className="flex flex-wrap gap-2">
            {sortOptions.map((option) => {
              const isActive = sortBy === option.key;
              return (
                <Button
                  key={option.key}
                  variant={isActive ? "default" : "outline"}
                  size="sm"
                  onClick={() => {
                    if (isActive) {
                      setSortDirection(
                        sortDirection === "asc" ? "desc" : "asc"
                      );
                    } else {
                      setSortBy(option.key);
                      setSortDirection(option.key === "name" ? "asc" : "desc");
                    }
                  }}
                >
                  {option.label}
                </Button>
              );
            })}
          </div>
        </div>
        <p className="text-muted-foreground text-xs">
          Showing {filteredAndSortedItems.length} of {allInstitutions.length}
        </p>
        {joinError && <p className="text-destructive text-sm">{joinError}</p>}
      </header>

      <main className="grid flex-1 grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredAndSortedItems.map((institution) => {
          const mode: Mode = membershipsSet.has(institution.id)
            ? "member"
            : "directory";

          return (
            <InstitutionCard
              key={institution.id}
              institution={institution}
              mode={mode}
              isJoining={isJoining}
              onClick={handleOpenInstitution}
              onJoin={handleJoin}
              onManage={(inst) => setSelectedInstitution(inst)}
            />
          );
        })}
      </main>

      {filteredAndSortedItems.length === 0 && (
        <div className="border-border bg-card text-muted-foreground rounded-xl border p-8 text-center">
          No institutions found.
        </div>
      )}

      <InstitutionCreateForm
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
      />

      {selectedInstitution && (
        <InstitutionManagementModal
          isOpen
          onClose={() => setSelectedInstitution(null)}
          institution={selectedInstitution}
          currentUserRole={selectedInstitution.currentUserRole}
        />
      )}
    </div>
  );
};
