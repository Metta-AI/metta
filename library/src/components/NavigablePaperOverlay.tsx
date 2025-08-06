"use client";

import React from "react";
import {
  PaperWithUserContext,
  User,
  UserInteraction,
} from "@/posts/data/papers";
import { useOverlayNavigation } from "./OverlayStack";
import { loadAuthorClient, AuthorDTO } from "@/posts/data/authors-client";

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
      let fullAuthor = await loadAuthorClient(authorId);

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
        const fallbackAuthor: AuthorDTO = {
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
    <div className="max-h-[90vh] max-w-4xl overflow-hidden rounded-lg bg-white shadow-xl">
      <div className="max-h-[90vh] space-y-6 overflow-y-auto p-6">
        {/* Header with title and close button */}
        <div className="flex items-start justify-between gap-4">
          <h1 className="flex-1 text-xl leading-tight font-semibold text-gray-900">
            {paper.title}
          </h1>
          <button
            onClick={onClose}
            className="mt-1 flex-shrink-0 rounded-lg p-2 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
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
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
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

        {/* Abstract */}
        {paper.abstract && (
          <div>
            <h3 className="mb-3 text-sm font-medium text-gray-700">Abstract</h3>
            <p className="text-lg leading-relaxed text-gray-900">
              {paper.abstract}
            </p>
          </div>
        )}

        {/* Topic tags */}
        {paper.tags && paper.tags.length > 0 && (
          <div>
            <h3 className="mb-2 text-sm font-medium text-gray-700">Topics</h3>
            <div className="flex flex-wrap gap-2">
              {paper.tags.map((tag, index) => (
                <span
                  key={index}
                  className="inline-block rounded-full bg-gray-100 px-3 py-1 text-sm text-gray-700"
                >
                  {tag}
                </span>
              ))}
            </div>
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

        {/* Actions */}
        <div className="flex items-center gap-4 border-t border-gray-200 pt-4">
          <button
            onClick={() => onStarToggle(paper.id)}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 transition-colors ${
              paper.isStarredByCurrentUser
                ? "bg-yellow-100 text-yellow-800 hover:bg-yellow-200"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            <svg
              className="h-5 w-5"
              fill={paper.isStarredByCurrentUser ? "currentColor" : "none"}
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z"
              />
            </svg>
            {paper.isStarredByCurrentUser ? "Starred" : "Star"}
          </button>

          <button
            onClick={() => onQueueToggle(paper.id)}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 transition-colors ${
              paper.isQueuedByCurrentUser
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
            {paper.isQueuedByCurrentUser ? "In Queue" : "Add to Queue"}
          </button>
        </div>

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
      </div>
    </div>
  );
}
