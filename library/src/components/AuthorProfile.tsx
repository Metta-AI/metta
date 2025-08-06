"use client";

import { FC, useState } from "react";

import { AuthorDTO } from "@/posts/data/authors-client";

interface AuthorProfileProps {
  author: AuthorDTO;
  onClose?: () => void;
  onInstitutionClick?: (institutionName: string) => void;
}

/**
 * AuthorProfile Component
 *
 * Displays detailed information about a single author including their profile,
 * statistics, papers, and network information in a tabbed interface.
 * Designed to work as an overlay/modal.
 */
export const AuthorProfile: FC<AuthorProfileProps> = ({
  author,
  onClose,
  onInstitutionClick,
}) => {
  const [activeTab, setActiveTab] = useState<"overview" | "papers" | "network">(
    "overview"
  );
  const [isFollowing, setIsFollowing] = useState(false);

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const formatDate = (date: Date | string) => {
    // Convert string to Date if needed
    const dateObj = typeof date === "string" ? new Date(date) : date;

    // Check if the date is valid
    if (isNaN(dateObj.getTime())) {
      return "Invalid date";
    }

    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    }).format(dateObj);
  };

  const formatRelativeDate = (date: Date | string | null) => {
    if (!date) return "Unknown";

    // Convert string to Date if needed
    const dateObj = typeof date === "string" ? new Date(date) : date;

    // Check if the date is valid
    if (isNaN(dateObj.getTime())) {
      return "Unknown";
    }

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

  const toggleFollow = () => {
    setIsFollowing(!isFollowing);
  };

  return (
    <div className="bg-white">
      {/* Header */}
      <div className="border-b border-gray-200">
        <div className="px-6 py-6">
          {/* Close Button */}
          {onClose && (
            <div className="mb-4 flex justify-end">
              <button
                onClick={onClose}
                className="p-2 text-gray-400 transition-colors hover:text-gray-600"
                aria-label="Close"
              >
                <svg
                  className="h-6 w-6"
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
          )}

          {/* Author Info */}
          <div className="flex items-start gap-6">
            <div className="bg-primary-500 flex h-24 w-24 flex-shrink-0 items-center justify-center rounded-full text-3xl font-semibold text-white">
              {author.avatar || getInitials(author.name)}
            </div>
            <div className="min-w-0 flex-1">
              <h2 className="mb-2 text-3xl font-bold text-gray-900">
                {author.name}
              </h2>
              {author.title && (
                <p className="mb-1 text-xl text-gray-600">{author.title}</p>
              )}
              {author.institution && (
                <div className="mb-3">
                  {onInstitutionClick ? (
                    <button
                      onClick={() => onInstitutionClick(author.institution!)}
                      className="text-left text-lg text-blue-600 transition-colors hover:text-blue-700 hover:underline"
                    >
                      {author.institution}
                    </button>
                  ) : (
                    <p className="text-lg text-gray-500">
                      {author.institution}
                    </p>
                  )}
                </div>
              )}

              <div className="mb-4 flex items-center gap-4">
                <div className="flex items-center gap-6 text-sm text-gray-600">
                  <div>
                    <span className="text-lg font-semibold text-gray-900">
                      {author.hIndex || 0}
                    </span>
                    <span className="ml-1">h-index</span>
                  </div>
                  <div>
                    <span className="text-lg font-semibold text-gray-900">
                      {(author.totalCitations || 0).toLocaleString()}
                    </span>
                    <span className="ml-1">citations</span>
                  </div>
                  <div>
                    <span className="text-lg font-semibold text-gray-900">
                      {author.paperCount}
                    </span>
                    <span className="ml-1">papers</span>
                  </div>
                </div>
                <button
                  onClick={toggleFollow}
                  className={`rounded-full px-6 py-2 font-medium transition-colors ${
                    isFollowing
                      ? "bg-gray-200 text-gray-700 hover:bg-gray-300"
                      : "bg-primary-500 hover:bg-primary-600 text-white"
                  }`}
                >
                  {isFollowing ? "Following" : "Follow"}
                </button>
                <span
                  className={`rounded-full px-3 py-1 text-sm font-semibold ${
                    author.claimed
                      ? "border border-green-200 bg-green-100 text-green-700"
                      : "border border-gray-200 bg-gray-100 text-gray-600"
                  }`}
                >
                  {author.claimed ? "Claimed Profile" : "Unclaimed Profile"}
                </span>
              </div>

              {author.expertise.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {author.expertise.map((exp, index) => (
                    <span
                      key={index}
                      className="rounded-full bg-gray-100 px-3 py-1 text-sm text-gray-700"
                    >
                      {exp}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <div className="px-6">
          <div className="flex space-x-8">
            {[
              { id: "overview", label: "Overview" },
              { id: "papers", label: `Papers (${author.recentPapers.length})` },
              { id: "network", label: "Network" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`border-b-2 px-1 py-4 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? "border-primary-500 text-primary-600"
                    : "border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="px-6 py-6">
        {activeTab === "overview" && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <div className="rounded-lg border border-gray-200 bg-white p-6">
                <h3 className="mb-4 text-lg font-semibold text-gray-900">
                  Recent Papers
                </h3>
                <div className="space-y-4">
                  {author.recentPapers.slice(0, 5).map((paper) => (
                    <div
                      key={paper.id}
                      className="border-b border-gray-100 pb-4 last:border-b-0"
                    >
                      <h4 className="mb-1 font-medium text-gray-900">
                        {paper.title}
                      </h4>
                      <p className="mb-2 text-sm text-gray-600">
                        {formatDate(paper.createdAt)} • {paper.stars} stars
                      </p>
                      {paper.link && (
                        <a
                          href={paper.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary-500 hover:text-primary-600 text-sm underline"
                        >
                          View Paper
                        </a>
                      )}
                    </div>
                  ))}
                  {author.recentPapers.length === 0 && (
                    <p className="text-sm text-gray-500">
                      No papers found for this author.
                    </p>
                  )}
                </div>
              </div>
            </div>
            <div className="space-y-6">
              <div className="rounded-lg border border-gray-200 bg-white p-6">
                <h3 className="mb-4 text-lg font-semibold text-gray-900">
                  Activity
                </h3>
                <p className="text-sm text-gray-600">
                  Last active {formatRelativeDate(author.recentActivity)}
                </p>
              </div>

              {author.orcid && (
                <div className="rounded-lg border border-gray-200 bg-white p-6">
                  <h3 className="mb-4 text-lg font-semibold text-gray-900">
                    External Profiles
                  </h3>
                  <div className="space-y-2">
                    <a
                      href={`https://orcid.org/${author.orcid}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary-500 hover:text-primary-600 block text-sm underline"
                    >
                      ORCID Profile
                    </a>
                    {author.googleScholarId && (
                      <a
                        href={`https://scholar.google.com/citations?user=${author.googleScholarId}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary-500 hover:text-primary-600 block text-sm underline"
                      >
                        Google Scholar
                      </a>
                    )}
                    {author.arxivId && (
                      <a
                        href={`https://arxiv.org/a/${author.arxivId}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary-500 hover:text-primary-600 block text-sm underline"
                      >
                        arXiv Profile
                      </a>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "papers" && (
          <div className="rounded-lg border border-gray-200 bg-white">
            <div className="border-b border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900">
                All Papers
              </h3>
            </div>
            <div className="divide-y divide-gray-200">
              {author.recentPapers.map((paper) => (
                <div key={paper.id} className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="mb-2 text-lg font-medium text-gray-900">
                        {paper.title}
                      </h4>
                      <div className="mb-3 flex items-center gap-4 text-sm text-gray-600">
                        <span>{formatDate(paper.createdAt)}</span>
                        <span>•</span>
                        <span>{paper.stars} stars</span>
                      </div>
                    </div>
                    {paper.link && (
                      <a
                        href={paper.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary-500 hover:text-primary-600 ml-4"
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
                            d="M14 3h7v7m0 0L10 21l-7-7 11-11z"
                          />
                        </svg>
                      </a>
                    )}
                  </div>
                </div>
              ))}
              {author.recentPapers.length === 0 && (
                <div className="p-6 text-center text-gray-500">
                  No papers found for this author.
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "network" && (
          <div className="rounded-lg border border-gray-200 bg-white p-6">
            <h3 className="mb-4 text-lg font-semibold text-gray-900">
              Collaboration Network
            </h3>
            <p className="text-gray-600">
              Network visualization coming soon...
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
