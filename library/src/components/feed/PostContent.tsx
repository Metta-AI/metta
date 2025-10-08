"use client";

import { RichTextRenderer } from "@/components/RichTextRenderer";
import { cleanArxivUrls } from "@/lib/utils/url";

interface PostContentProps {
  content: string;
}

export function PostContent({ content }: PostContentProps) {
  const cleanedContent = cleanArxivUrls(content).trim();

  if (!cleanedContent) return null;

  return (
    <div className="px-4 pb-2 text-[14px] leading-[1.55] whitespace-pre-wrap text-neutral-900">
      <RichTextRenderer text={cleanedContent} />
    </div>
  );
}
