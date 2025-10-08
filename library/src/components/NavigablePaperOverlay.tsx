"use client";

import React, { useState, useEffect } from "react";
import {
  PaperWithUserContext,
  User,
  UserInteraction,
} from "@/posts/data/papers";
import { useOverlayNavigation } from "./OverlayStack";
import * as authorsApi from "@/lib/api/resources/authors";
import type { AuthorDetail } from "@/lib/api/resources/authors";
import { StarWidgetQuery } from "./StarWidgetQuery";
import { BaseOverlay } from "@/components/overlays/BaseOverlay";

interface NavigablePaperOverlayProps {
  paper: PaperWithUserContext;
  users: User[];
  interactions: UserInteraction[];
  onClose: () => void;
  onStarToggle: (paperId: string) => void;
  onQueueToggle: (paperId: string) => void;
}

export default function NavigablePaperOverlay({
  paper,
  users,
  interactions,
  onClose,
  onStarToggle,
  onQueueToggle,
}: NavigablePaperOverlayProps) {
  const { openAuthor, openInstitution } = useOverlayNavigation();

  // Local state for optimistic updates
  const [localPaper, setLocalPaper] = useState(paper);

  // Update local state when paper prop changes (only if it's actually different)
  useEffect(() => {
    if (localPaper.id !== paper.id) {
      setLocalPaper(paper);
    }
  }, [paper, localPaper.id]);

  // Handle optimistic queue toggle
  const handleQueueToggle = () => {
    // Optimistically update local state
    setLocalPaper((prev) => ({
      ...prev,
      isQueuedByCurrentUser: !prev.isQueuedByCurrentUser,
    }));

    // Call the parent handler
    onQueueToggle(paper.id);
  };

  // Get interactions for this paper
  const paperInteractions = interactions.filter((i) => i.paperId === paper.id);

  // Get users who have interacted with this paper
  const usersWithInteractions = users.filter((user) =>
    paperInteractions.some((interaction) => interaction.userId === user.id)
  );

  // Get users by interaction type
  const starredUsers = usersWithInteractions.filter((user) =>
    paperInteractions.some((i) => i.userId === user.id && i.starred)
  );
  const queuedUsers = usersWithInteractions.filter((user) =>
    paperInteractions.some((i) => i.userId === user.id && i.queued)
  );
  const readUsers = usersWithInteractions.filter((user) =>
    paperInteractions.some((i) => i.userId === user.id && i.readAt)
  );

  // Handle clicking on an author
  const handleAuthorClick = async (authorId: string, authorName: string) => {
    try {
      // Try to load full author data by ID first
      let fullAuthor = await authorsApi.getAuthor(authorId);

      // If that fails, try searching by name via the authors API
      if (!fullAuthor) {
        try {
          const searchResponse = await fetch(
            `/api/authors?search=${encodeURIComponent(authorName)}`
          );
          if (searchResponse.ok) {
            const searchResults = await searchResponse.json();
            if (searchResults.length > 0) {
              // Use the first matching author
              fullAuthor = searchResults[0];
            }
          }
        } catch (searchError) {
          console.log("Search by name failed:", searchError);
        }
      }

      if (fullAuthor) {
        openAuthor(fullAuthor);
      } else {
        // Fallback: create a minimal author object if all loading attempts fail
        const fallbackAuthor: AuthorDetail = {
          id: authorId,
          name: authorName,
          username: null,
          email: null,
          avatar: null,
          institution: null,
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
          paperCount: 0,
          recentPapers: [],
        };
        openAuthor(fallbackAuthor);
      }
    } catch (error) {
      console.error("Error loading author:", error);
    }
  };

  // Handle clicking on an institution
  const handleInstitutionClick = (institutionName: string) => {
    // For now, we'll open with just the name and empty arrays
    // In a real app, you'd fetch institution data from an API
    openInstitution(institutionName, [], []);
  };

  return (
    <BaseOverlay
      open
      onClose={onClose}
      title={paper.title}
      size="xl"
      contentClassName="space-y-6"
      footer={
        <button
          onClick={handleQueueToggle}
          className={`flex items-center gap-2 rounded-lg px-4 py-2 transition-colors ${
            localPaper.isQueuedByCurrentUser
              ? "bg-blue-100 text-blue-800 hover:bg-blue-200"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          <svg
            className="h-5 w-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 6v6m0 0v6m0-6h6m-6 0H6"
            />
          </svg>
          {localPaper.isQueuedByCurrentUser ? "In Queue" : "Add to Queue"}
        </button>
      }
    >
      {/* Header with star widget, title and close button */}
      <div className="flex items-start gap-3">
        <div className="mt-1 flex-shrink-0">
          <StarWidgetQuery
            paperId={paper.id}
            initialTotalStars={starredUsers.length}
            initialIsStarredByCurrentUser={localPaper.isStarredByCurrentUser}
            size="md"
          />
        </div>
      </div>

      {/* Metadata grid */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {/* Authors - now clickable */}
        <div>
          <h3 className="mb-2 text-sm font-medium text-gray-700">Authors</h3>
          <div className="space-y-1">
            {paper.authors && paper.authors.length > 0 ? (
              paper.authors.map((author, index) => (
                <button
                  key={author.id || index}
                  onClick={() => handleAuthorClick(author.id, author.name)}
                  className="block text-left text-blue-600 transition-colors hover:text-blue-800 hover:underline"
                  title={`View ${author.name}'s profile`}
                >
                  {author.name}
                </button>
              ))
            ) : (
              <p className="text-gray-500 italic">No authors listed</p>
            )}
          </div>
        </div>

        {/* Institutions - now clickable */}
        <div>
          <h3 className="mb-2 text-sm font-medium text-gray-700">
            Institutions
          </h3>
          <div className="space-y-1">
            {paper.institutions && paper.institutions.length > 0 ? (
              paper.institutions.map((institution, index) => (
                <button
                  key={index}
                  onClick={() => handleInstitutionClick(institution)}
                  className="block text-left text-blue-600 transition-colors hover:text-blue-800 hover:underline"
                  title={`View papers from ${institution}`}
                >
                  {institution}
                </button>
              ))
            ) : (
              <p className="text-gray-500 italic">No institutions listed</p>
            )}
          </div>
        </div>
      </div>

      {/* Tags */}
      {paper.tags && paper.tags.length > 0 && (
        <div className="mb-6">
          <h3 className="mb-3 text-sm font-medium text-gray-700">Tags</h3>
          <div className="flex flex-wrap gap-2">
            {paper.tags.map((tag, index) => (
              <button
                key={index}
                onClick={() => {
                  const params = new URLSearchParams();
                  params.set("search", tag);
                  window.open(`/papers?${params.toString()}`, "_blank");
                }}
                className="inline-block cursor-pointer rounded px-3 py-1 text-sm font-bold transition-colors hover:bg-gray-200"
                style={{ backgroundColor: "#EFF3F9", color: "#131720" }}
                title={`Click to view all papers tagged with "${tag}"`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Abstract */}
      {paper.abstract && (
        <div>
          <h3 className="mb-3 text-sm font-medium text-gray-700">Abstract</h3>
          <p className="text-lg leading-relaxed text-gray-900">
            {paper.abstract}
          </p>
        </div>
      )}

      {/* Links */}
      {paper.link && (
        <div>
          <h3 className="mb-2 text-sm font-medium text-gray-700">Links</h3>
          <a
            href={paper.link}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            View Paper
          </a>
        </div>
      )}

      {/* User interactions */}
      {(starredUsers.length > 0 ||
        queuedUsers.length > 0 ||
        readUsers.length > 0) && (
        <div className="border-t border-gray-200 pt-4">
          <h3 className="mb-3 text-sm font-medium text-gray-700">
            Community Activity
          </h3>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {/* Starred by */}
            {starredUsers.length > 0 && (
              <div>
                <h4 className="mb-1 text-xs font-medium text-gray-600">
                  Starred by
                </h4>
                <div className="flex flex-wrap gap-1">
                  {starredUsers.map((user) => (
                    <span
                      key={user.id}
                      className="rounded bg-yellow-100 px-2 py-1 text-xs text-yellow-800"
                    >
                      {user.name || user.email}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Queued by */}
            {queuedUsers.length > 0 && (
              <div>
                <h4 className="mb-1 text-xs font-medium text-gray-600">
                  Queued by
                </h4>
                <div className="flex flex-wrap gap-1">
                  {queuedUsers.map((user) => (
                    <span
                      key={user.id}
                      className="rounded bg-blue-100 px-2 py-1 text-xs text-blue-800"
                    >
                      {user.name || user.email}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Read by */}
            {readUsers.length > 0 && (
              <div>
                <h4 className="mb-1 text-xs font-medium text-gray-600">
                  Read by
                </h4>
                <div className="flex flex-wrap gap-1">
                  {readUsers.map((user) => (
                    <span
                      key={user.id}
                      className="rounded bg-green-100 px-2 py-1 text-xs text-green-800"
                    >
                      {user.name || user.email}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </BaseOverlay>
  );
}
