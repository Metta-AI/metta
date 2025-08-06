"use client";

import { useState } from "react";
import {
  PaperWithUserContext,
  User,
  UserInteraction,
} from "@/posts/data/papers";
import { toggleStarAction } from "@/posts/actions/toggleStarAction";
import { toggleQueueAction } from "@/posts/actions/toggleQueueAction";
import { useOverlayNavigation } from "./OverlayStack";
import { signOut } from "next-auth/react";

interface MeViewProps {
  user: {
    id: string;
    name: string | null;
    email: string | null;
    image: string | null;
  };
  starredPapers: PaperWithUserContext[];
  queuedPapers: PaperWithUserContext[];
  allPapers: PaperWithUserContext[];
  users: User[];
  interactions: UserInteraction[];
}

export function MeView({
  user,
  starredPapers,
  queuedPapers,
  allPapers,
  users,
  interactions,
}: MeViewProps) {
  // Overlay navigation
  const { openPaper } = useOverlayNavigation();

  // Handle toggle star
  const handleToggleStar = async (paperId: string) => {
    try {
      const formData = new FormData();
      formData.append("paperId", paperId);
      await toggleStarAction(formData);

      // Local state update is now handled by the overlay stack system
    } catch (error) {
      console.error("Error toggling star:", error);
    }
  };

  // Handle toggle queue
  const handleToggleQueue = async (paperId: string) => {
    try {
      const formData = new FormData();
      formData.append("paperId", paperId);
      await toggleQueueAction(formData);

      // Local state update is now handled by the overlay stack system
    } catch (error) {
      console.error("Error toggling queue:", error);
    }
  };

  // Handle paper click
  const handlePaperClick = (paper: PaperWithUserContext) => {
    openPaper(paper, users, interactions, handleToggleStar, handleToggleQueue);
  };

  // Handle remove from starred
  const handleRemoveFromStarred = async (paperId: string) => {
    await handleToggleStar(paperId);
  };

  // Handle remove from queued
  const handleRemoveFromQueued = async (paperId: string) => {
    await handleToggleQueue(paperId);
  };

  // Handle sign out
  const handleSignOut = async () => {
    try {
      await signOut({ callbackUrl: "/" });
    } catch (error) {
      console.error("Error signing out:", error);
    }
  };

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

  return (
    <div className="mx-auto max-w-4xl p-6">
      {/* User Profile Section */}
      <div className="mb-8">
        <div className="rounded-lg border border-gray-200 bg-white p-6">
          <div className="flex items-center gap-4">
            {/* Profile Circle */}
            <div className="flex h-16 w-16 flex-shrink-0 items-center justify-center rounded-full bg-blue-600 text-xl font-semibold text-white">
              {getUserInitials(user.name, user.email)}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-700">Name:</span>
                <span className="text-lg font-semibold text-gray-900">
                  {user.name || "Unknown User"}
                </span>
              </div>
              <div className="mt-1 flex items-center gap-2">
                <span className="text-sm font-medium text-gray-700">
                  Email:
                </span>
                <span className="text-lg text-gray-900">{user.email}</span>
              </div>
            </div>
            <button
              onClick={handleSignOut}
              className="rounded-lg bg-gray-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-gray-700"
            >
              Sign Out
            </button>
          </div>
        </div>
      </div>

      {/* Starred Papers Section */}
      <div className="mb-8">
        <h2 className="mb-3 text-xl font-bold text-gray-900">
          Starred ({starredPapers.length})
        </h2>
        {starredPapers.length > 0 ? (
          <div className="divide-y divide-gray-200 rounded-lg border border-gray-200 bg-white">
            {starredPapers.map((paper) => (
              <div key={paper.id} className="p-3 hover:bg-gray-50">
                <div className="flex items-center justify-between">
                  <div className="min-w-0 flex-1">
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
                  <button
                    onClick={() => handleRemoveFromStarred(paper.id)}
                    className="ml-3 rounded-lg p-1.5 text-gray-400 transition-colors hover:bg-red-50 hover:text-red-600"
                    title="Remove from starred"
                  >
                    <svg
                      className="h-4 w-4"
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
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-gray-200 bg-white p-4 text-center">
            <p className="text-gray-500">No starred papers yet.</p>
          </div>
        )}
      </div>

      {/* Queued Papers Section */}
      <div className="mb-8">
        <h2 className="mb-3 text-xl font-bold text-gray-900">
          Queued ({queuedPapers.length})
        </h2>
        {queuedPapers.length > 0 ? (
          <div className="divide-y divide-gray-200 rounded-lg border border-gray-200 bg-white">
            {queuedPapers.map((paper) => (
              <div key={paper.id} className="p-3 hover:bg-gray-50">
                <div className="flex items-center justify-between">
                  <div className="min-w-0 flex-1">
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
                  <button
                    onClick={() => handleRemoveFromQueued(paper.id)}
                    className="ml-3 rounded-lg p-1.5 text-gray-400 transition-colors hover:bg-red-50 hover:text-red-600"
                    title="Remove from queue"
                  >
                    <svg
                      className="h-4 w-4"
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
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-gray-200 bg-white p-4 text-center">
            <p className="text-gray-500">No queued papers yet.</p>
          </div>
        )}
      </div>
    </div>
  );
}
