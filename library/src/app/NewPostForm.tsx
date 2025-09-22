"use client";
import { useAction } from "next-safe-action/hooks";
import { FC, useState } from "react";

import { createPostAction } from "@/posts/actions/createPostAction";

/**
 * NewPostForm Component
 *
 * Allows users to create new posts with rich content including:
 * - Post content with LaTeX support via MathJax
 * - Automatic paper detection from URLs
 * - Simple, clean interface matching the mockup
 */
export const NewPostForm: FC = () => {
  const [content, setContent] = useState("");
  const [error, setError] = useState<string | null>(null);

  const { execute, isExecuting } = useAction(createPostAction, {
    onSuccess: () => {
      // Reset form and error
      setContent("");
      setError(null);
      // The feed is paginated, and paginated state is stored on the
      // client side only. So refreshing the entire page is the easiest way to
      // update the list of posts.
      window.location.reload();
    },
    onError: (error) => {
      console.error("Error creating post:", error);
      // Extract error message from the error object
      const serverError = error.error?.serverError;
      const validationErrors = error.error?.validationErrors;
      const errorMessage =
        (typeof serverError === "object" &&
        serverError !== null &&
        "message" in serverError
          ? (serverError as any).message
          : null) ||
        (Array.isArray(validationErrors) &&
        validationErrors.length > 0 &&
        typeof validationErrors[0] === "object" &&
        validationErrors[0] !== null &&
        "message" in validationErrors[0]
          ? (validationErrors[0] as any).message
          : null) ||
        "Failed to create post. Please try again.";
      setError(errorMessage);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!content.trim()) return;

    // Clear any previous errors
    setError(null);

    const formData = new FormData();
    formData.append("title", "New Post"); // Default title
    formData.append("content", content);

    execute(formData);
  };

  return (
    <div className="border-b border-gray-200 bg-white p-6">
      <div className="flex gap-3">
        <textarea
          className="max-h-32 min-h-[96px] flex-1 resize-none rounded-lg border border-gray-200 px-4 py-3 text-sm leading-relaxed text-gray-900 placeholder-gray-400 focus:border-transparent focus:ring-2 focus:ring-blue-500"
          placeholder={`Share your thoughts about a paper...\nMust include an arXiv URL (e.g., https://arxiv.org/abs/2301.12345)\nLaTeX supported: $x^2 + y^2 = z^2$ for inline, $$\\alpha + \\beta = \\gamma$$ for display`}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onKeyDown={(e) => {
            // Submit on Command+Enter (Mac) or Ctrl+Enter (Windows/Linux)
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              handleSubmit(e);
            }
            // Allow Enter to create newlines by not preventing default
          }}
        />

        {/* Post button */}
        <button
          className={`self-end rounded-lg px-6 py-2 text-sm font-medium transition-colors ${
            content.trim() && !isExecuting
              ? "bg-blue-600 text-white hover:bg-blue-700"
              : "cursor-not-allowed bg-gray-200 text-gray-500"
          }`}
          disabled={!content.trim() || isExecuting}
          onClick={handleSubmit}
        >
          {isExecuting ? "Posting..." : "Post"}
        </button>
      </div>

      {/* Error message */}
      {error && (
        <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-red-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">
                Error creating post
              </h3>
              <p className="mt-1 text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
