"use client";

import { FC, useState, useRef, useEffect } from "react";
import { FeedPostDTO } from "@/posts/data/feed";
import { LLMAbstract } from "@/lib/llm-abstract-generator-clean";
import { useOverlayNavigation } from "@/components/OverlayStack";
import { StarWidget } from "@/components/StarWidget";
import { toggleStarAction } from "@/posts/actions/toggleStarAction";

interface PaperSidebarProps {
  paper: FeedPostDTO["paper"];
}

interface LLMAbstractViewProps {
  llmAbstract: LLMAbstract;
  originalAbstract?: string | null;
  pdfUrl?: string;
  homepageUrl?: string;
}

/**
 * PaperSidebar Component
 *
 * Shows paper overview information in the right sidebar including:
 * - Paper title and metadata
 * - Full abstract (expanded)
 * - Authors and institutions
 * - Tags and external link
 */
const InstitutionsSection: FC<{ institutions: string[] }> = ({
  institutions,
}) => {
  const { openInstitution } = useOverlayNavigation();

  const handleInstitutionClick = (institutionName: string) => {
    openInstitution(institutionName, [], []);
  };

  return (
    <div className="mb-6">
      <h4 className="mb-3 font-semibold text-gray-900">Institutions</h4>
      <div className="space-y-1">
        {institutions.map((institution, index) => (
          <button
            key={index}
            onClick={() => handleInstitutionClick(institution)}
            className="cursor-pointer text-left text-sm text-blue-600 underline hover:text-blue-700"
          >
            {institution}
          </button>
        ))}
      </div>
    </div>
  );
};

export const PaperSidebar: FC<PaperSidebarProps> = ({ paper }) => {
  // Local state for optimistic star updates
  const [localPaperData, setLocalPaperData] = useState(paper);

  // Only update local state if the paper actually changed (not just re-rendered)
  useEffect(() => {
    if (!localPaperData || localPaperData.id !== paper?.id) {
      setLocalPaperData(paper);
    }
  }, [paper, localPaperData]);

  // Handle star toggle
  const handleToggleStar = async () => {
    if (!localPaperData) return;

    try {
      // Optimistic update
      setLocalPaperData((prev) =>
        prev
          ? {
              ...prev,
              starred: !prev.starred,
              stars: prev.starred ? prev.stars - 1 : prev.stars + 1,
            }
          : null
      );

      // Call the server action
      await toggleStarAction({ paperId: localPaperData.id });
    } catch (error) {
      console.error("Failed to toggle star:", error);
      // Revert optimistic update on error
      setLocalPaperData(paper);
    }
  };

  if (!localPaperData) {
    return (
      <div className="h-full w-full overflow-y-auto border-l border-gray-200 bg-gray-50 p-6">
        <div className="text-center text-gray-500">
          <p className="text-sm">No paper associated with this post</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full overflow-y-auto border-l border-gray-200 bg-white p-6">
      {/* Paper Header */}
      <div className="mb-6">
        <h2 className="mb-2 text-lg font-semibold text-gray-900">
          Paper Overview
        </h2>

        {/* Paper Title */}
        <h3 className="mb-4 text-xl leading-tight font-bold text-gray-900">
          {paper.source === "arxiv" && paper.externalId ? (
            <a
              href={`https://arxiv.org/abs/${paper.externalId}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 underline transition-colors hover:text-blue-700"
            >
              {paper.title}
            </a>
          ) : (
            paper.title
          )}
        </h3>

        {/* Paper Metadata */}
        <div className="space-y-2 text-sm text-gray-600">
          {paper.source && (
            <div className="flex items-center gap-2">
              <span className="font-medium">Source:</span>
              {paper.link ? (
                <a
                  href={paper.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 transition-colors hover:text-blue-700"
                >
                  {paper.source}
                </a>
              ) : (
                <span>{paper.source}</span>
              )}
            </div>
          )}

          {paper.externalId && (
            <div className="flex items-center gap-2">
              <span className="font-medium">ID:</span>
              <span className="rounded bg-gray-100 px-2 py-1 font-mono text-xs">
                {paper.externalId}
              </span>
            </div>
          )}

          <div className="flex items-center gap-2">
            <span className="font-medium">Stars:</span>
            <StarWidget
              totalStars={localPaperData.stars}
              isStarredByCurrentUser={localPaperData.starred}
              onClick={handleToggleStar}
              size="xl"
            />
          </div>
        </div>
      </div>

      {/* Authors */}
      {paper.authors && paper.authors.length > 0 && (
        <div className="mb-6">
          <h4 className="mb-3 font-semibold text-gray-900">Authors</h4>
          <div className="space-y-2">
            {paper.authors.map((author) => (
              <div key={author.id} className="text-sm">
                <div className="font-medium text-gray-900">{author.name}</div>
                {author.institution && (
                  <div className="text-xs text-gray-600">
                    {author.institution}
                  </div>
                )}
                {author.orcid && (
                  <div className="font-mono text-xs text-gray-500">
                    ORCID: {author.orcid}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Institutions */}
      {paper.institutions && paper.institutions.length > 0 && (
        <InstitutionsSection institutions={paper.institutions} />
      )}

      {/* Tags */}
      {paper.tags && paper.tags.length > 0 && (
        <div className="mb-6">
          <h4 className="mb-3 font-semibold text-gray-900">Tags</h4>
          <div className="flex flex-wrap gap-2">
            {paper.tags.map((tag, index) => (
              <span
                key={index}
                className="inline-block rounded-full bg-blue-100 px-3 py-1 text-sm font-medium text-blue-800"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Enhanced Abstract or Original Abstract */}
      {paper.llmAbstract ? (
        <LLMAbstractView
          llmAbstract={paper.llmAbstract as LLMAbstract}
          originalAbstract={paper.abstract}
          pdfUrl={
            paper.source === "arxiv" && paper.externalId
              ? `https://arxiv.org/pdf/${paper.externalId}.pdf`
              : paper.link || undefined
          }
          homepageUrl={
            paper.source === "arxiv" && paper.externalId
              ? `https://arxiv.org/abs/${paper.externalId}`
              : undefined
          }
        />
      ) : paper.abstract ? (
        <div className="mb-6">
          <h4 className="mb-3 font-semibold text-gray-900">Abstract</h4>
          <div className="text-sm leading-relaxed whitespace-pre-wrap text-gray-700">
            {paper.abstract}
          </div>
        </div>
      ) : null}

      {/* Timestamps */}
      <div className="mt-6 border-t border-gray-200 pt-4">
        <div className="space-y-1 text-xs text-gray-500">
          <div>
            <span className="font-medium">Created:</span>{" "}
            {new Date(paper.createdAt).toLocaleDateString()}
          </div>
          <div>
            <span className="font-medium">Updated:</span>{" "}
            {new Date(paper.updatedAt).toLocaleDateString()}
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Component to display LLM-generated enhanced abstract
 */
const LLMAbstractView: FC<LLMAbstractViewProps> = ({
  llmAbstract,
  originalAbstract,
  pdfUrl,
  homepageUrl,
}) => {
  const [activeTab, setActiveTab] = useState<"summary" | "abstract">(
    "summary" // Default to summary when LLM abstract is available
  );

  return (
    <div className="mb-6">
      {/* Header with links */}
      <div className="mb-4 flex items-center justify-end">
        <div className="flex gap-2">
          {pdfUrl && (
            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-blue-600 underline hover:text-blue-700"
            >
              PDF
            </a>
          )}
          {homepageUrl && (
            <a
              href={homepageUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-blue-600 underline hover:text-blue-700"
            >
              Homepage
            </a>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="mb-4 border-b border-gray-200">
        <div className="flex space-x-4">
          <button
            onClick={() => setActiveTab("summary")}
            className={`border-b-2 pb-2 text-sm font-medium transition-colors ${
              activeTab === "summary"
                ? "border-blue-500 text-blue-600"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            AI Summary
          </button>
          {originalAbstract && (
            <button
              onClick={() => setActiveTab("abstract")}
              className={`border-b-2 pb-2 text-sm font-medium transition-colors ${
                activeTab === "abstract"
                  ? "border-blue-500 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              Original Abstract
            </button>
          )}
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === "summary" && (
        <div className="space-y-6">
          {/* Overview (formerly Paper Summary) */}
          <div>
            <h5 className="mb-2 text-sm font-medium text-gray-900">Overview</h5>
            <div className="rounded bg-gray-50 p-3 text-sm leading-relaxed text-gray-700">
              {llmAbstract.summary}
            </div>
          </div>

          {/* Key Figures with Images */}
          {llmAbstract.figuresWithImages.length > 0 && (
            <div>
              <h5 className="mb-3 text-sm font-medium text-gray-900">
                Key Figures ({llmAbstract.figuresWithImages.length})
              </h5>
              <div className="space-y-6">
                {llmAbstract.figuresWithImages.map((figure, index) => (
                  <div key={index} className="border-l-4 border-blue-200 pl-4">
                    <div className="mb-2 flex items-start justify-between gap-2">
                      <div>
                        <h6 className="text-sm font-medium text-gray-900">
                          {figure.figureNumber}
                        </h6>
                        <div className="text-xs text-gray-500">
                          Page {figure.pageNumber} • Confidence:{" "}
                          {(figure.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>

                    <div className="mb-2 text-sm font-medium text-gray-800">
                      {figure.caption}
                    </div>

                    <div className="mb-3 text-sm text-gray-700">
                      {figure.explanation}
                    </div>

                    {figure.significance && (
                      <div className="mb-3 rounded bg-blue-50 p-2 text-xs text-blue-800">
                        <strong>Why it's important:</strong>{" "}
                        {figure.significance}
                      </div>
                    )}

                    {figure.imageData && (
                      <div className="mt-3">
                        <img
                          src={`data:image/${figure.imageType || "png"};base64,${figure.imageData}`}
                          alt={figure.caption}
                          className="h-auto max-w-full cursor-pointer rounded border border-gray-200 shadow-sm transition-shadow hover:shadow-md"
                          style={{ maxHeight: "400px" }}
                          onClick={(e) => {
                            // Create a downloadable blob URL for better viewing
                            const byteCharacters = atob(figure.imageData!);
                            const byteNumbers = new Array(
                              byteCharacters.length
                            );
                            for (let i = 0; i < byteCharacters.length; i++) {
                              byteNumbers[i] = byteCharacters.charCodeAt(i);
                            }
                            const byteArray = new Uint8Array(byteNumbers);
                            const blob = new Blob([byteArray], {
                              type: `image/${figure.imageType || "png"}`,
                            });
                            const url = URL.createObjectURL(blob);
                            window.open(url, "_blank");
                            // Clean up the URL after a delay
                            setTimeout(() => URL.revokeObjectURL(url), 1000);
                          }}
                          title="Click to open in new tab"
                        />
                        <div className="mt-1 text-xs text-gray-500">
                          Click to view full size •{" "}
                          {figure.imageType?.toUpperCase() || "PNG"}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Page Count */}
          <div className="text-xs text-gray-500">
            Total pages: {llmAbstract.pageCount}
          </div>
        </div>
      )}

      {activeTab === "abstract" && originalAbstract && (
        <div className="space-y-4">
          <div>
            <h5 className="mb-2 text-sm font-medium text-gray-900">
              Original Abstract
            </h5>
            <div className="text-sm leading-relaxed whitespace-pre-wrap text-gray-700">
              {originalAbstract}
            </div>
          </div>
        </div>
      )}

      {/* Generation Info */}
      <div className="mt-4 border-t border-gray-200 pt-4">
        <div className="text-xs text-gray-500">
          Enhanced abstract generated on{" "}
          {new Date(llmAbstract.generatedAt).toLocaleDateString()}
        </div>
      </div>
    </div>
  );
};
