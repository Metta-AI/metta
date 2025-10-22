"use client";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { FC, useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { useSearchParams } from "next/navigation";
import { Paperclip, X, Image as ImageIcon } from "lucide-react";
import { z } from "zod";

import { useCreatePost } from "@/hooks/mutations/useCreatePost";
import { MentionInput } from "@/components/MentionInput";
import { QuotedPostPreview } from "@/components/QuotedPostPreview";
import { parseMentions } from "@/lib/mentions";
import { useErrorHandling } from "@/lib/hooks/useErrorHandling";
import type { FeedPostDTO } from "@/posts/data/feed";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { ErrorAlert } from "@/components/ui/error-alert";

const formSchema = z.object({
  content: z
    .string()
    .min(1, "Content is required")
    .max(10_000, "Post content is too long"),
});

type FormValues = z.infer<typeof formSchema>;

type AttachedImage = {
  id: string;
  url: string;
  file?: File;
  filename: string;
  size: number;
};

type PendingQuoteDraft = {
  content: string;
  quotedPostIds: string[];
};

const draftStorageKey = "quote-draft";

interface NewPostFormProps {
  onPostCreated?: (post: FeedPostDTO) => void;
}

export const NewPostForm: FC<NewPostFormProps> = ({ onPostCreated }) => {
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: { content: "" },
  });
  const { control, handleSubmit, reset, setValue, watch } = form;
  const content = watch("content");

  const [attachedImages, setAttachedImages] = useState<AttachedImage[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [mentions, setMentions] = useState<string[]>([]);
  const [quotedPostIds, setQuotedPostIds] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const searchParams = useSearchParams();

  const {
    error: submitError,
    setError: setSubmitError,
    clearError: clearSubmitError,
  } = useErrorHandling({
    fallbackMessage: "Failed to create post. Please try again.",
  });

  const createPostMutation = useCreatePost({
    onPostCreated,
  });

  const uploadImageMutation = useMutation<AttachedImage, Error, File>({
    mutationFn: async (file: File) => {
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
        id: Math.random().toString(36).slice(2, 11),
        url: result.imageUrl,
        filename: result.filename,
        size: result.size,
        file,
      } satisfies AttachedImage;
    },
    onMutate: () => {
      setIsUploading(true);
    },
    onSuccess: (image) => {
      setAttachedImages((prev) => [...prev, image]);
      setIsUploading(false);
    },
    onError: (error) => {
      setSubmitError(error);
      setIsUploading(false);
    },
  });

  useEffect(() => {
    const quoteParam = searchParams.get("quote");
    if (!quoteParam) return;

    const quoteDraft = sessionStorage.getItem(draftStorageKey);
    if (!quoteDraft) return;

    try {
      const parsed: PendingQuoteDraft = JSON.parse(quoteDraft);
      if (parsed.content) {
        setValue("content", parsed.content, { shouldDirty: true });
      }
      if (parsed.quotedPostIds) {
        setQuotedPostIds(parsed.quotedPostIds);
      }
    } catch (error) {
      console.error("Error parsing quote draft:", error);
    } finally {
      sessionStorage.removeItem(draftStorageKey);
    }
  }, [searchParams, setValue]);

  const updateMentionsFromContent = (rawContent: string) => {
    const parsedMentions = parseMentions(rawContent);
    const mentionValues = parsedMentions.map((m) => m.raw);
    setMentions(mentionValues);
  };

  const updateQuotedPostIds = (rawContent: string) => {
    import("@/lib/post-link-parser").then(({ extractPostIdsFromContent }) => {
      const detectedPostIds = extractPostIdsFromContent(rawContent);

      if (detectedPostIds.length > 0) {
        setQuotedPostIds((current) => {
          const combined = [...current];
          for (const postId of detectedPostIds) {
            if (!combined.includes(postId) && combined.length < 2) {
              combined.push(postId);
            }
          }
          return combined;
        });
      } else if (quotedPostIds.length > 0) {
        const remainingIds = quotedPostIds.filter(
          (id) => rawContent.includes(id) || rawContent.includes(`/posts/${id}`)
        );
        if (remainingIds.length !== quotedPostIds.length) {
          setQuotedPostIds(remainingIds);
        }
      }
    });
  };

  const handleContentChange = (nextValue: string) => {
    setValue("content", nextValue, { shouldDirty: true, shouldValidate: true });
    updateMentionsFromContent(nextValue);
    if (nextValue) {
      updateQuotedPostIds(nextValue);
    } else {
      setMentions([]);
      setQuotedPostIds([]);
    }
  };

  const handleRemoveQuotedPost = (postId: string) => {
    setQuotedPostIds((prev) => prev.filter((id) => id !== postId));
    const currentContent = form.getValues("content") ?? "";
    const urlPattern = new RegExp(
      `https?:\\/\\/[^\\s]*\\/posts\\/${postId}|^\\/posts\\/${postId}`,
      "gi"
    );
    const nextContent = currentContent.replace(urlPattern, "").trim();
    setValue("content", nextContent, {
      shouldDirty: true,
      shouldValidate: true,
    });
    updateMentionsFromContent(nextContent);
  };

  const handleFileSelect = async (files: FileList | null) => {
    if (!files) return;

    const imageFiles = Array.from(files).filter((file) =>
      file.type.startsWith("image/")
    );

    if (imageFiles.length === 0) {
      setSubmitError("Please select valid image files");
      return;
    }

    for (const file of imageFiles) {
      uploadImageMutation.mutate(file);
    }
  };

  const handlePaste = async (event: React.ClipboardEvent) => {
    const items = Array.from(event.clipboardData.items);
    const imageItems = items.filter((item) => item.type.startsWith("image/"));

    if (imageItems.length === 0) return;

    event.preventDefault();

    for (const item of imageItems) {
      const file = item.getAsFile();
      if (file) {
        uploadImageMutation.mutate(file);
      }
    }
  };

  const removeImage = (imageId: string) => {
    setAttachedImages((prev) => prev.filter((img) => img.id !== imageId));
  };

  const onSubmit = (values: FormValues) => {
    if (!values.content.trim()) return;

    clearSubmitError();
    updateMentionsFromContent(values.content);

    const generateTitle = (content: string): string => {
      const cleanContent = content
        .trim()
        .replace(/\n/g, " ")
        .replace(/\s+/g, " ");

      if (cleanContent.length <= 50) {
        return cleanContent;
      }

      const truncated = cleanContent.substring(0, 47);
      const lastSpace = truncated.lastIndexOf(" ");
      return (
        (lastSpace > 20 ? truncated.substring(0, lastSpace) : truncated) + "..."
      );
    };

    const title = generateTitle(values.content);

    createPostMutation.mutate(
      {
        title,
        content: values.content,
        images: attachedImages.map((img) => img.url),
        mentions,
        quotedPostIds,
      },
      {
        onSuccess: () => {
          reset({ content: "" });
          clearSubmitError();
          setAttachedImages([]);
          setQuotedPostIds([]);
        },
        onError: (error) => {
          setSubmitError(error);
        },
      }
    );
  };

  return (
    <Card className="border-b border-gray-200 bg-white p-4 md:p-6">
      <Form {...form}>
        <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-3">
          <div className="flex flex-col gap-3 sm:flex-row">
            <FormField
              control={control}
              name="content"
              render={({ field }) => (
                <FormItem className="flex-1">
                  <FormLabel className="sr-only">Post content</FormLabel>
                  <FormControl>
                    <MentionInput
                      wrapperClassName="flex-1"
                      className="max-h-32 min-h-[96px] resize-none rounded-lg border border-gray-200 px-3 py-2 text-sm leading-relaxed text-gray-900 placeholder-gray-400 focus:border-transparent focus:ring-2 focus:ring-blue-500 md:px-4 md:py-3"
                      placeholder={`Share your thoughts about a paper...\nMust include an arXiv URL (e.g., https://arxiv.org/abs/2301.12345)\nLaTeX supported: $x^2 + y^2 = z^2$ for inline, $$\\alpha + \\beta = \\gamma$$ for display\nPaste images or attach files below.\n\nTry @-mentioning: @username for users, @/groupname for your institution's groups, @domain.com/groupname for specific institution groups.`}
                      value={field.value}
                      onChange={(value) => {
                        field.onChange(value);
                        handleContentChange(value);
                      }}
                      onMentionsChange={(values) => setMentions(values)}
                      onPaste={handlePaste}
                      onKeyDown={(event) => {
                        if (
                          event.key === "Enter" &&
                          (event.metaKey || event.ctrlKey)
                        ) {
                          event.preventDefault();
                          void handleSubmit(onSubmit)();
                        }
                      }}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <div className="flex justify-end sm:flex-col sm:gap-2">
              <Button
                type="submit"
                disabled={
                  !content.trim() || isUploading || createPostMutation.isPending
                }
                className="rounded-lg px-4 py-2 text-sm font-medium transition-colors md:px-6"
              >
                {createPostMutation.isPending
                  ? "Posting..."
                  : isUploading
                    ? "Uploading..."
                    : "Post"}
              </Button>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={(event) => handleFileSelect(event.target.files)}
              />

              <Button
                type="button"
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-1"
              >
                <Paperclip className="h-4 w-4" /> Attach
              </Button>

              {isUploading && (
                <span className="text-sm text-blue-600">Uploading...</span>
              )}
            </div>

            {attachedImages.length > 0 && (
              <span className="text-sm text-gray-500">
                {attachedImages.length} image
                {attachedImages.length !== 1 ? "s" : ""} attached
              </span>
            )}
          </div>

          {quotedPostIds.length > 0 && (
            <QuotedPostPreview
              quotedPostIds={quotedPostIds}
              onRemove={handleRemoveQuotedPost}
            />
          )}

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

          {submitError && (
            <ErrorAlert
              title="Error creating post"
              message={submitError}
              onDismiss={clearSubmitError}
              className="mt-3"
            />
          )}
        </form>
      </Form>
    </Card>
  );
};
