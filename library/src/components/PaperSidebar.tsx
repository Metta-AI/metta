"use client";

import { FC, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Download } from "lucide-react";
import { FeedPostDTO } from "@/posts/data/feed";
import { LLMAbstract } from "@/lib/llm-abstract-generator-clean";
import { useOverlayNavigation } from "@/components/OverlayStack";
import { StarWidgetQuery } from "@/components/StarWidgetQuery";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/Button";

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

export const PaperSidebar: FC<PaperSidebarProps> = ({ paper }) => {
  const router = useRouter();

  // Handle tag click to navigate to papers view with tag filter
  const handleTagClick = (tag: string) => {
    const params = new URLSearchParams();
    params.set("search", tag);
    router.push(`/papers?${params.toString()}`);
  };

  if (!paper) {
    return (
      <div className="h-screen flex-1 overflow-y-auto border-l bg-neutral-50">
        <div className="px-4 py-4">
          <div className="text-center text-neutral-500">
            <p className="text-sm">No paper associated with this post</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex-1 overflow-y-auto border-l bg-white">
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
          <div className="pl-2">
            {(paper.source === "arxiv" && paper.externalId) || paper.link ? (
              <Button
                size="small"
                theme="default"
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
                <a key={author.id} href="#" className="inline-block">
                  <Badge variant="secondary" className="rounded-md">
                    {author.name}
                  </Badge>
                </a>
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
                <a key={index} href="#" className="inline-block">
                  <Badge variant="secondary" className="rounded-md">
                    {institution}
                  </Badge>
                </a>
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

          {/* Key Figures */}
          {llmAbstract.figuresWithImages.length > 0 && (
            <div>
              <div className="mb-1 text-[12px] font-semibold text-neutral-700">
                Key Figures ({llmAbstract.figuresWithImages.length})
              </div>
              <div className="space-y-3">
                {llmAbstract.figuresWithImages.map((figure, index) => (
                  <figure key={index} className="rounded-xl border p-3">
                    <div className="flex items-center gap-2 text-[12px] text-neutral-700">
                      <span className="font-semibold">
                        {figure.figureNumber}
                      </span>
                      <span>• Page {figure.pageNumber}</span>
                      <span>
                        • Confidence: {Math.round(figure.confidence * 100)}%
                      </span>
                    </div>
                    <div className="mt-2 text-[13px] font-medium text-neutral-900">
                      {figure.caption}
                    </div>
                    {/* Preview placeholder or actual image */}
                    {figure.imageData ? (
                      <div className="mt-2">
                        <img
                          src={`data:image/${figure.imageType || "png"};base64,${figure.imageData}`}
                          alt={figure.caption}
                          className="h-24 w-full cursor-pointer rounded-lg object-cover"
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
                      </div>
                    ) : (
                      <div className="mt-2 grid h-24 place-content-center rounded-lg bg-neutral-100 text-[12px] text-neutral-500">
                        preview
                      </div>
                    )}
                    <figcaption className="mt-2 text-[13px] text-neutral-900">
                      {figure.explanation}
                    </figcaption>
                    {figure.significance && (
                      <div className="mt-2 rounded-lg border border-blue-200 bg-blue-50 p-2 text-[12.5px] text-blue-900">
                        <span className="font-semibold">
                          Why it's important:{" "}
                        </span>
                        {figure.significance}
                      </div>
                    )}
                  </figure>
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
