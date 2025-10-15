"use client";

import { FC, useRef, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Download } from "lucide-react";
import { FeedPostDTO } from "@/posts/data/feed";
import { LLMAbstract } from "@/lib/llm-abstract-generator-clean";
import { useOverlayNavigation } from "@/components/OverlayStack";
import { StarWidgetQuery } from "@/components/StarWidgetQuery";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useAuthor, useAuthors } from "@/hooks/queries";

interface PaperSidebarProps {
  paper: FeedPostDTO["paper"];
  onClose?: () => void;
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

export const PaperSidebar: FC<PaperSidebarProps> = ({ paper, onClose }) => {
  const router = useRouter();
  const { openAuthor, openInstitution } = useOverlayNavigation();
  const [selectedAuthorId, setSelectedAuthorId] = useState<string | null>(null);
  const [searchName, setSearchName] = useState<string | null>(null);

  // Fetch author by ID when selected
  const { data: authorById } = useAuthor(selectedAuthorId!, {
    enabled: !!selectedAuthorId,
  });

  // Fallback: search by name if ID fetch failed
  const { data: authorsByName } = useAuthors(
    { search: searchName! },
    { enabled: !!searchName && !authorById }
  );

  // Open author when data is loaded
  useEffect(() => {
    if (authorById) {
      openAuthor(authorById);
      setSelectedAuthorId(null);
      setSearchName(null);
    } else if (authorsByName && authorsByName.length > 0) {
      openAuthor(authorsByName[0]);
      setSelectedAuthorId(null);
      setSearchName(null);
    }
  }, [authorById, authorsByName, openAuthor]);

  // Handle tag click to navigate to papers view with tag filter
  const handleTagClick = (tag: string) => {
    const params = new URLSearchParams();
    params.set("search", tag);
    router.push(`/papers?${params.toString()}`);
  };

  // Handle clicking on an author - trigger reactive fetch
  const handleAuthorClick = (authorId: string, authorName: string) => {
    setSelectedAuthorId(authorId);
    setSearchName(authorName); // Fallback if ID doesn't work
  };

  // Handle clicking on an institution
  const handleInstitutionClick = (institutionName: string) => {
    openInstitution(institutionName, [], []);
  };

  if (!paper) {
    return (
      <div className="h-full overflow-y-auto bg-neutral-50 md:h-screen md:w-[55%] md:flex-shrink-0 md:border-l">
        <div className="px-4 py-4">
          <div className="text-center text-neutral-500">
            <p className="text-sm">No paper associated with this post</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-white md:h-screen md:w-[55%] md:flex-shrink-0 md:border-l">
      {/* Mobile header with close button */}
      <div className="flex flex-shrink-0 items-center justify-between border-b border-gray-200 bg-white px-4 py-3 shadow-sm md:hidden">
        <h2 className="text-lg font-semibold text-gray-900">Paper Details</h2>
        {onClose && (
          <button
            onClick={onClose}
            className="cursor-pointer flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-700"
            title="Close paper details"
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
        )}
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="space-y-4 px-4 py-4">
          {/* Header row: star + title + download */}
          <div className="flex items-start gap-2.5">
            <div className="mt-0.5">
              <StarWidgetQuery
                paperId={paper.id}
                initialTotalStars={paper.stars}
                initialIsStarredByCurrentUser={paper.starred}
                size="sm"
              />
            </div>
            <div className="min-w-0 flex-1">
              {paper.source === "arxiv" && paper.externalId ? (
                <a
                  href={`https://arxiv.org/abs/${paper.externalId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[15.5px] leading-[1.3] font-semibold tracking-tight text-neutral-900 hover:underline"
                >
                  {paper.title}
                </a>
              ) : (
                <div className="text-[15.5px] leading-[1.3] font-semibold tracking-tight text-neutral-900">
                  {paper.title}
                </div>
              )}
              <div className="mt-1 flex items-center gap-2 text-[12.5px] text-neutral-600">
                {paper.source === "arxiv" && paper.externalId ? (
                  <a
                    href={`https://arxiv.org/abs/${paper.externalId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline"
                  >
                    arXiv
                  </a>
                ) : (
                  "Unknown Venue"
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 pl-2">
              {(paper.source === "arxiv" && paper.externalId) || paper.link ? (
                <Button
                  size="sm"
                  variant="default"
                  onClick={() => {
                    const pdfUrl =
                      paper.source === "arxiv" && paper.externalId
                        ? `https://arxiv.org/pdf/${paper.externalId}.pdf`
                        : paper.link;
                    if (pdfUrl) {
                      const filename =
                        paper.source === "arxiv" && paper.externalId
                          ? `${paper.externalId}.pdf`
                          : `${paper.title.replace(/[^a-z0-9]/gi, "_").toLowerCase()}.pdf`;

                      // Use our API endpoint to proxy the PDF download
                      const downloadUrl = `/api/download-pdf?url=${encodeURIComponent(pdfUrl)}&filename=${encodeURIComponent(filename)}`;

                      // Create a temporary anchor element to trigger download
                      const link = document.createElement("a");
                      link.href = downloadUrl;
                      link.download = filename;
                      document.body.appendChild(link);
                      link.click();
                      document.body.removeChild(link);
                    }
                  }}
                  type="button"
                >
                  <Download className="h-4 w-4" />
                </Button>
              ) : null}

              {/* Close button - Desktop only */}
              {onClose && (
                <button
                  onClick={onClose}
                  className="cursor-pointer hidden rounded-lg p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600 md:block"
                  title="Close paper details"
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
              )}
            </div>
          </div>

          <hr className="border-neutral-200" />

          {/* Authors as clickable chips */}
          {paper.authors && paper.authors.length > 0 && (
            <section>
              <div className="mb-1 text-[12px] font-semibold text-neutral-700">
                Authors
              </div>
              <div className="flex flex-wrap gap-1">
                {paper.authors.map((author) => (
                  <button
                    key={author.id}
                    onClick={() => handleAuthorClick(author.id, author.name)}
                    className="inline-block cursor-pointer"
                  >
                    <Badge
                      variant="secondary"
                      className="rounded-md text-[11px] transition-colors hover:bg-neutral-200"
                    >
                      {author.name}
                    </Badge>
                  </button>
                ))}
              </div>
            </section>
          )}

          {/* Institutions as clickable chips */}
          {paper.institutions && paper.institutions.length > 0 && (
            <section>
              <div className="mb-1 text-[12px] font-semibold text-neutral-700">
                Institutions
              </div>
              <div className="flex flex-wrap gap-1">
                {paper.institutions.map((institution, index) => (
                  <button
                    key={index}
                    onClick={() => handleInstitutionClick(institution)}
                    className="inline-block cursor-pointer"
                  >
                    <Badge
                      variant="secondary"
                      className="rounded-md text-[11px] transition-colors hover:bg-neutral-200"
                    >
                      {institution}
                    </Badge>
                  </button>
                ))}
              </div>
            </section>
          )}

          {/* Topic tags */}
          {paper.tags && paper.tags.length > 0 && (
            <section>
              <div className="mb-1 text-[12px] font-semibold text-neutral-700">
                Tags
              </div>
              <div className="flex flex-wrap gap-1">
                {paper.tags.map((tag, index) => (
                  <a
                    key={index}
                    href="#"
                    className="inline-block"
                    onClick={(e) => {
                      e.preventDefault();
                      handleTagClick(tag);
                    }}
                  >
                    <Badge variant="secondary" className="rounded-md">
                      {tag}
                    </Badge>
                  </a>
                ))}
              </div>
            </section>
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
            <section>
              <div className="mb-1 text-[12px] font-semibold text-neutral-700">
                Abstract
              </div>
              <div className="text-[13.5px] leading-[1.6] whitespace-pre-wrap text-neutral-800">
                {paper.abstract}
              </div>
            </section>
          ) : null}

          {/* Timestamps */}
          <div className="mt-6 border-t border-neutral-200 pt-4">
            <div className="space-y-1 text-xs text-neutral-500">
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

          {/* Spacer to ensure last content isn't flush to bottom */}
          <div className="h-8" />
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
    <div>
      {/* Tabs */}
      <nav className="text-[12.5px]">
        <div
          role="tablist"
          className="flex items-center gap-6 border-b border-neutral-200"
        >
          <button
            role="tab"
            aria-selected={activeTab === "summary"}
            className={
              activeTab === "summary"
                ? "-mb-px border-b-2 border-neutral-900 pb-2 font-medium text-neutral-900"
                : "-mb-px border-b-2 border-transparent pb-2 text-neutral-600 hover:text-neutral-800"
            }
            onClick={() => setActiveTab("summary")}
          >
            AI Summary
          </button>
          {originalAbstract && (
            <button
              role="tab"
              aria-selected={activeTab === "abstract"}
              className={
                activeTab === "abstract"
                  ? "-mb-px border-b-2 border-neutral-900 pb-2 font-medium text-neutral-900"
                  : "-mb-px border-b-2 border-transparent pb-2 text-neutral-600 hover:text-neutral-800"
              }
              onClick={() => setActiveTab("abstract")}
            >
              Original Abstract
            </button>
          )}
        </div>
      </nav>

      {/* Content */}
      {activeTab === "summary" ? (
        <section className="mt-4 space-y-3">
          {/* Overview */}
          <div>
            <div className="mb-1 text-[12px] font-semibold text-neutral-700">
              Overview
            </div>
            <p className="text-[13.5px] leading-[1.6] text-neutral-800">
              {llmAbstract.summary}
            </p>
          </div>

          {/* Figure Insights */}
          {llmAbstract.figuresWithImages &&
            llmAbstract.figuresWithImages.length > 0 && (
              <div>
                <div className="mb-2 text-[12px] font-semibold text-neutral-700">
                  Key Figures ({llmAbstract.figuresWithImages.length})
                </div>
                <div className="space-y-3">
                  {llmAbstract.figuresWithImages.map((figure, index) => (
                    <div
                      key={index}
                      className="rounded border border-neutral-200 bg-neutral-50 p-3"
                    >
                      <div className="mb-1 text-[12px] font-medium text-neutral-900">
                        {figure.figureNumber || `Figure ${index + 1}`}
                        {figure.pageNumber && (
                          <span className="ml-1 text-neutral-500">
                            (Page {figure.pageNumber})
                          </span>
                        )}
                      </div>

                      {figure.caption && (
                        <p className="mb-2 text-[12px] text-neutral-700">
                          <span className="font-medium">Caption:</span>{" "}
                          {figure.caption}
                        </p>
                      )}

                      {/* AI Commentary - Significance (Why it matters) */}
                      {(figure as any).significance && (
                        <p className="mb-2 text-[12px] leading-[1.5] text-neutral-800">
                          {(figure as any).significance}
                        </p>
                      )}

                      {/* Fallback to combined context if significance not available */}
                      {!(figure as any).significance &&
                        (figure as any).context && (
                          <p className="text-[12px] leading-[1.5] text-neutral-800">
                            {(figure as any).context}
                          </p>
                        )}
                    </div>
                  ))}
                </div>
              </div>
            )}
        </section>
      ) : (
        <section className="mt-4 text-[13.5px] leading-[1.6] text-neutral-800">
          {originalAbstract}
        </section>
      )}

      {/* Generation Info */}
      <div className="mt-4 border-t border-neutral-200 pt-4">
        <div className="text-xs text-neutral-500">
          AI summary generated using GPT-4 Vision from full PDF analysis on{" "}
          {new Date(llmAbstract.generatedAt).toLocaleDateString()}
        </div>
      </div>
    </div>
  );
};
