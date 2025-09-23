"use client";

import React from "react";
import { User, Users } from "lucide-react";
import { parseMentions, replaceMentionsInText } from "@/lib/mentions";

interface MentionTextProps {
  text: string;
  className?: string;
}

/**
 * Component that renders text with mentions styled as interactive elements
 */
export const MentionText: React.FC<MentionTextProps> = ({
  text,
  className = "",
}) => {
  // Parse mentions and replace them with styled spans
  const renderTextWithMentions = (inputText: string) => {
    const mentions = parseMentions(inputText);

    if (mentions.length === 0) {
      return inputText;
    }

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    mentions.forEach((mention, index) => {
      const mentionIndex = inputText.indexOf(mention.raw, lastIndex);

      // Add text before this mention
      if (mentionIndex > lastIndex) {
        parts.push(inputText.slice(lastIndex, mentionIndex));
      }

      // Add the styled mention
      parts.push(<MentionSpan key={`mention-${index}`} mention={mention} />);

      lastIndex = mentionIndex + mention.raw.length;
    });

    // Add remaining text after last mention
    if (lastIndex < inputText.length) {
      parts.push(inputText.slice(lastIndex));
    }

    return parts;
  };

  return <span className={className}>{renderTextWithMentions(text)}</span>;
};

interface MentionSpanProps {
  mention: {
    type: "user" | "group-relative" | "group-absolute";
    raw: string;
    value: string;
    domain?: string;
    groupName?: string;
    username?: string;
  };
}

const MentionSpan: React.FC<MentionSpanProps> = ({ mention }) => {
  const getMentionIcon = () => {
    switch (mention.type) {
      case "user":
        return <User className="h-3 w-3" />;
      case "group-relative":
      case "group-absolute":
        return <Users className="h-3 w-3" />;
      default:
        return null;
    }
  };

  const getMentionColor = () => {
    switch (mention.type) {
      case "user":
        return "text-blue-600 bg-blue-50 border-blue-200 hover:bg-blue-100";
      case "group-relative":
        return "text-green-600 bg-green-50 border-green-200 hover:bg-green-100";
      case "group-absolute":
        return "text-purple-600 bg-purple-50 border-purple-200 hover:bg-purple-100";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const getMentionTitle = () => {
    switch (mention.type) {
      case "user":
        return `User mention: ${mention.username}`;
      case "group-relative":
        return `Group mention: ${mention.groupName} (in your institution)`;
      case "group-absolute":
        return `Group mention: ${mention.groupName} (in ${mention.domain})`;
      default:
        return "Mention";
    }
  };

  return (
    <span
      className={`inline-flex cursor-pointer items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium transition-colors ${getMentionColor()}`}
      title={getMentionTitle()}
      onClick={() => {
        // TODO: Add click handlers for navigation
        console.log("Clicked mention:", mention);
      }}
    >
      {getMentionIcon()}
      {mention.raw}
    </span>
  );
};

