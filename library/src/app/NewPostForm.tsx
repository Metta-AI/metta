"use client";
import { useAction } from "next-safe-action/hooks";
import { FC, useState, useRef, useCallback } from "react";
import { Paperclip, X, Image as ImageIcon } from "lucide-react";

import { createPostAction } from "@/posts/actions/createPostAction";
import { MentionInput } from "@/components/MentionInput";
import { parseMentions } from "@/lib/mentions";

/**
 * NewPostForm Component
 *
 * Allows users to create new posts with rich content including:
 * - Post content with LaTeX support via MathJax
 * - Automatic paper detection from URLs
 * - Simple, clean interface matching the mockup
 */
interface AttachedImage {
  id: string;
  url: string;
  file?: File;
  filename: string;
  size: number;
}

export const NewPostForm: FC = () => {
  const [content, setContent] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [attachedImages, setAttachedImages] = useState<AttachedImage[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [mentions, setMentions] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handler for when mentions change
  const handleMentionsChange = useCallback((newMentions: string[]) => {
    setMentions(newMentions);
  }, []);

  // Handler for content changes that also updates mentions
  const handleContentChange = useCallback((newContent: string) => {
    setContent(newContent);

    // Parse mentions from content
    const parsedMentions = parseMentions(newContent);
    const mentionValues = parsedMentions.map((m) => m.raw);
    setMentions(mentionValues);
  }, []);

  const { execute, isExecuting } = useAction(createPostAction, {
    onSuccess: () => {
      // Reset form and error
      setContent("");
      setError(null);
      setAttachedImages([]);
      setMentions([]);
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

  // Upload image helper function
  const uploadImage = useCallback(
    async (file: File): Promise<AttachedImage | null> => {
      try {
        setIsUploading(true);
        const formData = new FormData();
        formData.append("image", file);

        const response = await fetch("/api/upload-image", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Failed to upload image");
        }

        const result = await response.json();

        return {
          id: Math.random().toString(36).substr(2, 9),
          url: result.imageUrl,
          filename: result.filename,
          size: result.size,
          file,
        };
      } catch (error) {
        console.error("Error uploading image:", error);
        setError(
          error instanceof Error ? error.message : "Failed to upload image"
        );
        return null;
      } finally {
        setIsUploading(false);
      }
    },
    []
  );

  // Handle file selection from button
  const handleFileSelect = useCallback(
    async (files: FileList | null) => {
      if (!files) return;

      const imageFiles = Array.from(files).filter((file) =>
        file.type.startsWith("image/")
      );

      if (imageFiles.length === 0) {
        setError("Please select valid image files");
        return;
      }

      for (const file of imageFiles) {
        const uploadedImage = await uploadImage(file);
        if (uploadedImage) {
          setAttachedImages((prev) => [...prev, uploadedImage]);
        }
      }
    },
    [uploadImage]
  );

  // Handle paste events for images
  const handlePaste = useCallback(
    async (e: React.ClipboardEvent) => {
      const items = Array.from(e.clipboardData.items);
      const imageItems = items.filter((item) => item.type.startsWith("image/"));

      if (imageItems.length === 0) return;

      e.preventDefault();

      for (const item of imageItems) {
        const file = item.getAsFile();
        if (file) {
          const uploadedImage = await uploadImage(file);
          if (uploadedImage) {
            setAttachedImages((prev) => [...prev, uploadedImage]);
          }
        }
      }
    },
    [uploadImage]
  );

  // Remove attached image
  const removeImage = useCallback((imageId: string) => {
    setAttachedImages((prev) => prev.filter((img) => img.id !== imageId));
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!content.trim()) return;

    // Clear any previous errors
    setError(null);

    const formData = new FormData();
    formData.append("title", "New Post"); // Default title
    formData.append("content", content);

    // Add images to form data
    if (attachedImages.length > 0) {
      formData.append(
        "images",
        JSON.stringify(attachedImages.map((img) => img.url))
      );
    }

    // Add mentions to form data
    if (mentions.length > 0) {
      formData.append("mentions", JSON.stringify(mentions));
    }

    execute(formData);
  };

  return (
    <div className="border-b border-gray-200 bg-white p-4 md:p-6">
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-3 sm:flex-row">
          <MentionInput
            wrapperClassName="flex-1"
            className="max-h-32 min-h-[96px] resize-none rounded-lg border border-gray-200 px-3 py-2 text-sm leading-relaxed text-gray-900 placeholder-gray-400 focus:border-transparent focus:ring-2 focus:ring-blue-500 md:px-4 md:py-3"
            placeholder={`Share your thoughts about a paper...\nMust include an arXiv URL (e.g., https://arxiv.org/abs/2301.12345)\nLaTeX supported: $x^2 + y^2 = z^2$ for inline, $$\\alpha + \\beta = \\gamma$$ for display\nPaste images or attach files below.\n\nTry @-mentioning: @username for users, @/groupname for your institution's groups, @domain.com/groupname for specific institution groups.`}
            value={content}
            onChange={handleContentChange}
            onMentionsChange={handleMentionsChange}
            onPaste={handlePaste}
            onKeyDown={(e) => {
              // Submit on Command+Enter (Mac) or Ctrl+Enter (Windows/Linux)
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                handleSubmit(e);
              }
              // Allow Enter to create newlines by not preventing default
            }}
          />

          <div className="flex justify-end sm:flex-col sm:gap-2">
            {/* Post button */}
            <button
              className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors md:px-6 ${
                content.trim() && !isExecuting && !isUploading
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "cursor-not-allowed bg-gray-200 text-gray-500"
              }`}
              disabled={!content.trim() || isExecuting || isUploading}
              onClick={handleSubmit}
            >
              {isExecuting
                ? "Posting..."
                : isUploading
                  ? "Uploading..."
                  : "Post"}
            </button>
          </div>
        </div>

        {/* Image attachment controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* File input (hidden) */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(e) => handleFileSelect(e.target.files)}
            />

            {/* Attach button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="flex items-center gap-1 rounded-lg border border-gray-200 px-3 py-1.5 text-sm text-gray-600 transition-colors hover:bg-gray-50 disabled:opacity-50"
              title="Attach images"
            >
              <Paperclip className="h-4 w-4" />
              Attach
            </button>

            {/* Upload status */}
            {isUploading && (
              <span className="text-sm text-blue-600">Uploading...</span>
            )}
          </div>

          {/* Image count */}
          {attachedImages.length > 0 && (
            <span className="text-sm text-gray-500">
              {attachedImages.length} image
              {attachedImages.length !== 1 ? "s" : ""} attached
            </span>
          )}
        </div>

        {/* Attached images preview */}
        {attachedImages.length > 0 && (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4">
            {attachedImages.map((image) => (
              <div key={image.id} className="group relative">
                <img
                  src={image.url}
                  alt={image.filename}
                  className="h-20 w-full rounded-lg border border-gray-200 object-cover"
                />
                <button
                  onClick={() => removeImage(image.id)}
                  className="absolute -top-1 -right-1 flex h-6 w-6 items-center justify-center rounded-full bg-red-500 text-white opacity-0 transition-opacity group-hover:opacity-100"
                  title="Remove image"
                >
                  <X className="h-3 w-3" />
                </button>
                <div
                  className="mt-1 truncate text-xs text-gray-500"
                  title={image.filename}
                >
                  {image.filename}
                </div>
              </div>
            ))}
          </div>
        )}

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
    </div>
  );
};
