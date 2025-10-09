"use client";

import React, { useState } from "react";

import type { AuthorDTO } from "@/posts/data/authors-client";
import type { PaperWithUserContext } from "@/posts/data/papers";
import { useOverlayNavigation } from "./OverlayStack";
import { useInstitution } from "@/hooks/queries";

interface InstitutionOverlayProps {
  institution: {
    name: string;
    papers: PaperWithUserContext[];
    authors: AuthorDTO[];
  };
  onClose: () => void;
}

export default function InstitutionOverlay({
  institution,
  onClose,
}: InstitutionOverlayProps) {
  const { openAuthor, openPaper } = useOverlayNavigation();
  const [activeTab, setActiveTab] = useState<"overview" | "papers" | "authors">(
    "overview"
  );

  // Fetch institution data if not provided - React Query handles caching and loading
  const shouldFetch =
    institution.papers.length === 0 && institution.authors.length === 0;
  const { data: fetchedData, isLoading: loading } = useInstitution(
    institution.name,
    {
      enabled: shouldFetch,
    }
  );

  // Use fetched data if available, otherwise use provided data
  const institutionPapers = fetchedData?.recentPapers || institution.papers;
  const institutionAuthors = fetchedData?.authors || institution.authors;

  const handleOpenPaper = (paper: any) => {
    // Only open paper if we have full PaperWithUserContext data
    if ("isStarredByCurrentUser" in paper) {
      openPaper(
        paper,
        [],
        [],
        () => {
          /** noop for institution context */
        },
        () => {
          /** noop for institution context */
        }
      );
    } else {
      // Paper data from API doesn't have full context, skip for now
      console.log("Paper click disabled - limited data from API");
    }
  };

  const handleOpenAuthor = (author: AuthorDTO) => {
    openAuthor(author);
  };

  const tabButtons = (
    <div className="flex gap-2">
      {(["overview", "papers", "authors"] as const).map((tab) => (
        <button
          key={tab}
          onClick={() => setActiveTab(tab)}
          className={`rounded-md px-3 py-1 text-sm font-medium ${
            activeTab === tab
              ? "bg-blue-50 text-blue-700"
              : "text-gray-600 hover:bg-gray-100"
          }`}
        >
          {tab === "papers"
            ? `Papers (${institutionPapers.length})`
            : tab === "authors"
              ? `Authors (${institutionAuthors.length})`
              : "Overview"}
        </button>
      ))}
    </div>
  );

  return (
    <div className="flex flex-col gap-6">
      {/* Institution description */}
      <div className="text-sm text-gray-600">
        {loading
          ? "Loading..."
          : `${institutionAuthors.length} authors • ${institutionPapers.length} papers`}
      </div>

      {tabButtons}

      <div className="flex-1 overflow-y-auto pr-1">
        {loading ? (
          <div className="flex items-center justify-center py-12 text-gray-500">
            Loading institution data...
          </div>
        ) : (
          <>
            {activeTab === "overview" && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">
                    Summary
                  </h3>
                  <p className="mt-2 text-sm text-gray-600">
                    Browse recent research and collaborators from{" "}
                    {institution.name}.
                  </p>
                </div>
                <div>
                  <h4 className="mb-3 text-sm font-semibold text-gray-800">
                    Highlighted Papers
                  </h4>
                  <div className="space-y-3">
                    {institutionPapers.slice(0, 3).map((paper: any) => (
                      <div
                        key={paper.id}
                        className="block w-full rounded-md border border-gray-200 bg-white p-4"
                      >
                        <div className="text-sm font-medium text-blue-700">
                          {paper.title}
                        </div>
                        <div className="mt-1 text-xs text-gray-500">
                          {paper.authors?.map((a: any) => a.name).join(", ") ||
                            "No authors"}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === "papers" && (
              <div className="space-y-4">
                {institutionPapers.length === 0 && (
                  <div className="rounded-md border border-dashed border-gray-300 p-6 text-center text-sm text-gray-500">
                    No papers available yet.
                  </div>
                )}
                {institutionPapers.map((paper: any) => (
                  <div
                    key={paper.id}
                    className="block w-full rounded-md border border-gray-200 bg-white p-4"
                  >
                    <div className="text-base font-semibold text-gray-900">
                      {paper.title}
                    </div>
                    <div className="mt-1 text-xs text-gray-500">
                      {paper.authors?.map((a: any) => a.name).join(", ") ||
                        "No authors"}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {activeTab === "authors" && (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {institutionAuthors.length === 0 && (
                  <div className="rounded-md border border-dashed border-gray-300 p-6 text-center text-sm text-gray-500">
                    No authors listed for this institution.
                  </div>
                )}
                {institutionAuthors.map((author: any) => (
                  <button
                    key={author.id}
                    onClick={() => handleOpenAuthor(author)}
                    className="flex items-center justify-between rounded-md border border-gray-200 bg-white p-4 text-left transition-colors hover:border-blue-200 hover:bg-blue-50"
                  >
                    <div>
                      <div className="text-sm font-semibold text-gray-900">
                        {author.name}
                      </div>
                      <div className="text-xs text-gray-500">
                        {author.paperCount} papers
                      </div>
                    </div>
                    <span className="text-sm text-blue-600">View</span>
                  </button>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
