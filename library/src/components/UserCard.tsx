"use client";

import { useState } from "react";
import {
  PaperWithUserContext,
  User,
  UserInteraction,
} from "@/posts/data/papers";
import { useStarMutation } from "@/hooks/useStarMutation";
import { toggleQueueAction } from "@/posts/actions/toggleQueueAction";
import PaperOverlay from "./PaperOverlay";

interface UserCardProps {
  user: User;
  allPapers: PaperWithUserContext[];
  users: User[];
  interactions: UserInteraction[];
  onClose: () => void;
}

export default function UserCard({
  user,
  allPapers,
  users,
  interactions,
  onClose,
}: UserCardProps) {
  // State for paper overlay
  const [selectedPaper, setSelectedPaper] =
    useState<PaperWithUserContext | null>(null);

  // Star mutation
  const starMutation = useStarMutation();

  // Get papers starred by this user
  const starredPapers = allPapers.filter((paper) => {
    const userInteraction = interactions.find(
      (i) => i.userId === user.id && i.paperId === paper.id && i.starred
    );
    return userInteraction !== undefined;
  });

  // Get papers queued by this user
  const queuedPapers = allPapers.filter((paper) => {
    const userInteraction = interactions.find(
      (i) => i.userId === user.id && i.paperId === paper.id && i.queued
    );
    return userInteraction !== undefined;
  });

  // Generate user initials for profile circle
  const getUserInitials = (name: string | null, email: string | null) => {
    if (name) {
      return name
        .split(" ")
        .map((n) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2);
    }
    if (email) {
      return email.charAt(0).toUpperCase();
    }
    return "?";
  };

  // Handle paper overlay close
  const handlePaperOverlayClose = () => {
    setSelectedPaper(null);
  };

  // Handle toggle star
  const handleToggleStar = (paperId: string) => {
    starMutation.mutate(paperId);
  };

  // Handle toggle queue
  const handleToggleQueue = async (paperId: string) => {
    try {
      const formData = new FormData();
      formData.append("paperId", paperId);
      await toggleQueueAction(formData);

      // Update local state immediately for better UX
      setSelectedPaper((prev) => {
        if (prev && prev.id === paperId) {
          return {
            ...prev,
            isQueuedByCurrentUser: !prev.isQueuedByCurrentUser,
          };
        }
        return prev;
      });
    } catch (error) {
      console.error("Error toggling queue:", error);
    }
  };

  // Handle paper click
  const handlePaperClick = (paper: PaperWithUserContext) => {
    setSelectedPaper(paper);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Semi-transparent backdrop */}
      <div
        className="absolute inset-0 bg-black/20 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* User card */}
      <div className="relative max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-xl bg-white shadow-2xl">
        {/* Content */}
        <div className="space-y-6 p-6">
          {/* Header with close button */}
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-4">
              {/* Profile Circle */}
              <div className="flex h-16 w-16 flex-shrink-0 items-center justify-center rounded-full bg-blue-600 text-xl font-semibold text-white">
                {getUserInitials(user.name, user.email)}
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  {user.name || "Unknown User"}
                </h1>
                <p className="text-lg text-gray-600">{user.email}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="flex-shrink-0 rounded-lg p-2 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
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

          {/* Starred Papers Section */}
          <div>
            <h2 className="mb-3 text-xl font-bold text-gray-900">
              Starred ({starredPapers.length})
            </h2>
            {starredPapers.length > 0 ? (
              <div className="divide-y divide-gray-200 rounded-lg border border-gray-200 bg-gray-50">
                {starredPapers.map((paper) => (
                  <div key={paper.id} className="p-3 hover:bg-gray-100">
                    <button
                      onClick={() => handlePaperClick(paper)}
                      className="w-full text-left transition-colors hover:text-blue-600"
                    >
                      <h3 className="truncate text-base font-medium text-gray-900">
                        {paper.title}
                      </h3>
                      {paper.authors && paper.authors.length > 0 && (
                        <p className="mt-1 text-sm text-gray-600">
                          {paper.authors
                            .map((author) => author.name)
                            .join(", ")}
                        </p>
                      )}
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-center">
                <p className="text-gray-500">No starred papers yet.</p>
              </div>
            )}
          </div>

          {/* Queued Papers Section */}
          <div>
            <h2 className="mb-3 text-xl font-bold text-gray-900">
              Queued ({queuedPapers.length})
            </h2>
            {queuedPapers.length > 0 ? (
              <div className="divide-y divide-gray-200 rounded-lg border border-gray-200 bg-gray-50">
                {queuedPapers.map((paper) => (
                  <div key={paper.id} className="p-3 hover:bg-gray-100">
                    <button
                      onClick={() => handlePaperClick(paper)}
                      className="w-full text-left transition-colors hover:text-blue-600"
                    >
                      <h3 className="truncate text-base font-medium text-gray-900">
                        {paper.title}
                      </h3>
                      {paper.authors && paper.authors.length > 0 && (
                        <p className="mt-1 text-sm text-gray-600">
                          {paper.authors
                            .map((author) => author.name)
                            .join(", ")}
                        </p>
                      )}
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-center">
                <p className="text-gray-500">No queued papers yet.</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Paper Overlay */}
      {selectedPaper && (
        <PaperOverlay
          paper={selectedPaper}
          users={users}
          interactions={interactions}
          onClose={handlePaperOverlayClose}
          onStarToggle={handleToggleStar}
          onQueueToggle={handleToggleQueue}
        />
      )}
    </div>
  );
}
