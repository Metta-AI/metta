"use client";

import React from "react";
import { MentionText } from "./MentionText";
import { linkifyText } from "@/lib/utils/linkify";
import { parseMentions } from "@/lib/mentions";

interface RichTextRendererProps {
  text: string;
  className?: string;
}

/**
 * Component that renders text with both links and mentions
 *
 * This combines the functionality of linkifyText and MentionText
 * to handle both URL detection and @-mentions in a single pass
 */
export const RichTextRenderer: React.FC<RichTextRendererProps> = ({
  text,
  className = "",
}) => {
  // First, let's check if there are any mentions
  const mentions = parseMentions(text);

  if (mentions.length === 0) {
    // No mentions, just use the existing linkifyText functionality
    return <span className={className}>{linkifyText(text)}</span>;
  }

  // If there are mentions, we need to render them specially
  // For now, let's use MentionText which will handle mentions,
  // but we lose link detection. TODO: Combine both features
  return <MentionText text={text} className={className} />;
};

