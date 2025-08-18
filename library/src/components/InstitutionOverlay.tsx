"use client";

import React, { useState, useEffect } from "react";
import { PaperWithUserContext } from "@/posts/data/papers";
import { AuthorDTO, loadAuthorClient } from "@/posts/data/authors-client";
import { useOverlayNavigation } from "./OverlayStack";
import { loadInstitutionClient } from "@/posts/data/institutions-client";

interface InstitutionData {
  name: string;
  papers: PaperWithUserContext[];
  authors: AuthorDTO[];
}

interface InstitutionOverlayProps {
  institution: InstitutionData;
  onClose: () => void;
}

/**
 * InstitutionOverlay
 *
 * Displays institution details including associated papers and authors.
 * Provides navigation to individual papers and authors from this institution.
 */
export default function InstitutionOverlay({
  institution,
  onClose,
}: InstitutionOverlayProps) {
  const { openAuthor, openPaper } = useOverlayNavigation();
  const [activeTab, setActiveTab] = useState<"overview" | "papers" | "authors">(
    "overview"
  );
  const [loading, setLoading] = useState(true);
  const [institutionPapers, setInstitutionPapers] = useState<
    PaperWithUserContext[]
  >([]);
  const [institutionAuthors, setInstitutionAuthors] = useState<AuthorDTO[]>([]);

  // Load institution data from API if not provided or if provided data is minimal
  useEffect(() => {
    const loadInstitutionData = async () => {
      setLoading(true);

      try {
        // If we have minimal or no data, fetch from API for complete institution data
        const shouldFetchFromAPI =
          !institution.papers.length ||
          !institution.authors.length ||
          institution.papers.length < 3; // Fetch if we have very few papers

        if (shouldFetchFromAPI) {
          console.log(
            "Fetching full institution data from API for:",
            institution.name
          );
          const fullInstitutionData = await loadInstitutionClient(
            institution.name
          );

          if (fullInstitutionData) {
            // Convert the API data to the expected format
            const apiPapers = fullInstitutionData.recentPapers.map((paper) => ({
              ...paper,
              abstract: null,
              institutions: [institution.name],
              tags: [],
              source: null,
              externalId: null,
              starred: null, // Add missing starred property
              isStarredByCurrentUser: false,
              isQueuedByCurrentUser: false,
              createdAt: new Date(paper.createdAt), // Ensure Date type
              updatedAt: new Date(),
            }));

            const apiAuthors = fullInstitutionData.authors.map(
              (authorData) => ({
                id: authorData.id,
                name: authorData.name,
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
                paperCount: authorData.paperCount,
                recentPapers: [],
              })
            );

            setInstitutionPapers(apiPapers);
            setInstitutionAuthors(apiAuthors);
          } else {
            // Fallback to provided data if API fails
            setInstitutionPapers(institution.papers || []);
            setInstitutionAuthors(institution.authors || []);
          }
        } else {
          // Use provided data if it seems complete
          setInstitutionPapers(institution.papers || []);
          setInstitutionAuthors(institution.authors || []);
        }
      } catch (error) {
        console.error("Error loading institution data:", error);
        // Fallback to provided data
        setInstitutionPapers(institution.papers || []);
        setInstitutionAuthors(institution.authors || []);
      }

      setLoading(false);
    };

    loadInstitutionData();
  }, [institution]);

  const handleAuthorClick = async (author: AuthorDTO) => {
    try {
      // Try to load full author data instead of using the simplified author object
      const fullAuthor = await loadAuthorClient(author.id);

      if (fullAuthor) {
        // Use the full author data which includes papers
        openAuthor(fullAuthor);
      } else {
        // Fallback to the simplified author if API fails
        console.log(
          `⚠️ Could not load full author data for ${author.name}, using simplified data`
        );
        openAuthor(author);
      }
    } catch (error) {
      console.error("Error loading full author data:", error);
      // Fallback to the simplified author
      openAuthor(author);
    }
  };

  const handlePaperClick = (paper: PaperWithUserContext) => {
    // Note: In a real implementation, you'd need to provide proper user interactions
    const mockUsers: any[] = [];
    const mockInteractions: any[] = [];
    const mockOnStarToggle = (paperId: string) =>
      console.log("Star toggle:", paperId);
    const mockOnQueueToggle = (paperId: string) =>
      console.log("Queue toggle:", paperId);

    openPaper(
      paper,
      mockUsers,
      mockInteractions,
      mockOnStarToggle,
      mockOnQueueToggle
    );
  };

  const getInstitutionInitials = (name: string) => {
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 3);
  };

  if (loading) {
    return (
      <div className="max-h-[90vh] max-w-4xl overflow-hidden rounded-lg bg-white shadow-xl">
        <div className="p-6">
          <div className="animate-pulse">
            <div className="mb-4 h-8 rounded bg-gray-200"></div>
            <div className="space-y-3">
              <div className="h-4 w-3/4 rounded bg-gray-200"></div>
              <div className="h-4 w-1/2 rounded bg-gray-200"></div>
              <div className="h-4 w-5/6 rounded bg-gray-200"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-h-[90vh] max-w-4xl overflow-hidden rounded-lg bg-white shadow-xl">
      <div className="max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="border-b border-gray-200">
          <div className="px-6 py-6">
            {/* Close Button */}
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

            {/* Institution Info */}
            <div className="flex items-start gap-6">
              <div className="flex h-24 w-24 flex-shrink-0 items-center justify-center rounded-lg bg-blue-600 text-2xl font-semibold text-white">
                {getInstitutionInitials(institution.name)}
              </div>
              <div className="min-w-0 flex-1">
                <h1 className="mb-2 text-3xl font-bold text-gray-900">
                  {institution.name}
                </h1>
                <div className="flex items-center gap-6 text-sm text-gray-600">
                  <span>{institutionPapers.length} papers</span>
                  <span>{institutionAuthors.length} authors</span>
                </div>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-gray-200">
            {[
              { key: "overview" as const, label: "Overview" },
              {
                key: "papers" as const,
                label: `Papers (${institutionPapers.length})`,
              },
              {
                key: "authors" as const,
                label: `Authors (${institutionAuthors.length})`,
              },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`border-b-2 px-6 py-3 text-sm font-medium transition-colors ${
                  activeTab === tab.key
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          {activeTab === "overview" && (
            <div className="space-y-6">
              <div>
                <h3 className="mb-4 text-lg font-semibold text-gray-900">
                  Institution Overview
                </h3>
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                  <div className="rounded-lg bg-gray-50 p-4">
                    <h4 className="mb-2 font-medium text-gray-900">
                      Research Output
                    </h4>
                    <p className="text-2xl font-bold text-blue-600">
                      {institutionPapers.length}
                    </p>
                    <p className="text-sm text-gray-600">Total Papers</p>
                  </div>
                  <div className="rounded-lg bg-gray-50 p-4">
                    <h4 className="mb-2 font-medium text-gray-900">
                      Research Community
                    </h4>
                    <p className="text-2xl font-bold text-green-600">
                      {institutionAuthors.length}
                    </p>
                    <p className="text-sm text-gray-600">Active Authors</p>
                  </div>
                </div>
              </div>

              {/* Recent Papers Preview */}
              {institutionPapers.length > 0 && (
                <div>
                  <h4 className="mb-3 font-medium text-gray-900">
                    Recent Papers
                  </h4>
                  <div className="space-y-2">
                    {institutionPapers.slice(0, 3).map((paper) => (
                      <button
                        key={paper.id}
                        onClick={() => handlePaperClick(paper)}
                        className="block w-full rounded-lg bg-gray-50 p-3 text-left transition-colors hover:bg-gray-100"
                      >
                        <p className="font-medium text-blue-600 hover:text-blue-800">
                          {paper.title}
                        </p>
                        {paper.authors && paper.authors.length > 0 && (
                          <p className="mt-1 text-sm text-gray-600">
                            {paper.authors
                              .map((author) => author.name)
                              .join(", ")}
                          </p>
                        )}
                      </button>
                    ))}
                    {institutionPapers.length > 3 && (
                      <button
                        onClick={() => setActiveTab("papers")}
                        className="text-sm font-medium text-blue-600 hover:text-blue-800"
                      >
                        View all {institutionPapers.length} papers →
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "papers" && (
            <div>
              <h3 className="mb-4 text-lg font-semibold text-gray-900">
                Papers
              </h3>
              {institutionPapers.length > 0 ? (
                <div className="space-y-4">
                  {institutionPapers.map((paper) => (
                    <div
                      key={paper.id}
                      className="rounded-lg border border-gray-200 p-4 transition-colors hover:border-gray-300"
                    >
                      <button
                        onClick={() => handlePaperClick(paper)}
                        className="block w-full text-left"
                      >
                        <h4 className="mb-2 font-medium text-blue-600 hover:text-blue-800">
                          {paper.title}
                        </h4>
                        {paper.authors && paper.authors.length > 0 && (
                          <p className="mb-2 text-sm text-gray-600">
                            Authors:{" "}
                            {paper.authors
                              .map((author) => author.name)
                              .join(", ")}
                          </p>
                        )}
                        {paper.tags && paper.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {paper.tags.slice(0, 5).map((tag, index) => (
                              <span
                                key={index}
                                className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-700"
                              >
                                {tag}
                              </span>
                            ))}
                            {paper.tags.length > 5 && (
                              <span className="text-xs text-gray-500">
                                +{paper.tags.length - 5} more
                              </span>
                            )}
                          </div>
                        )}
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
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
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <p className="text-gray-500">
                    No papers found for this institution.
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === "authors" && (
            <div>
              <h3 className="mb-4 text-lg font-semibold text-gray-900">
                Authors
              </h3>
              {institutionAuthors.length > 0 ? (
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  {institutionAuthors.map((author) => (
                    <button
                      key={author.id}
                      onClick={() => handleAuthorClick(author)}
                      className="rounded-lg border border-gray-200 p-4 text-left transition-colors hover:border-gray-300 hover:bg-gray-50"
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white">
                          {author.name
                            .split(" ")
                            .map((n) => n[0])
                            .join("")
                            .toUpperCase()
                            .slice(0, 2)}
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="truncate font-medium text-blue-600 hover:text-blue-800">
                            {author.name}
                          </p>
                          <p className="truncate text-sm text-gray-600">
                            {author.paperCount} papers
                          </p>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
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
                      d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z"
                    />
                  </svg>
                  <p className="text-gray-500">
                    No authors found for this institution.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
