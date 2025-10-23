"use client";

import React, {
  useState,
  useRef,
  useEffect,
  KeyboardEvent,
  ChangeEvent,
  TextareaHTMLAttributes,
} from "react";
import { Users, User, Building } from "lucide-react";

import {
  getMentionAtPosition,
  extractMentionQuery,
  MentionType,
} from "@/lib/mentions";
import * as mentionsApi from "@/lib/api/resources/mentions";

interface MentionSuggestion {
  type: MentionType;
  id: string;
  value: string;
  display: string;
  subtitle?: string;
  memberCount?: number;
}

interface MentionInputProps
  extends Omit<
    TextareaHTMLAttributes<HTMLTextAreaElement>,
    "onChange" | "onKeyDown"
  > {
  value: string;
  onChange: (value: string) => void;
  onMentionsChange?: (mentions: string[]) => void;
  wrapperClassName?: string; // For layout classes like flex-1
  onKeyDown?: (e: KeyboardEvent<HTMLTextAreaElement>) => void; // External keydown handler
}

export const MentionInput: React.FC<MentionInputProps> = ({
  value,
  onChange,
  onMentionsChange,
  className = "",
  wrapperClassName = "",
  onKeyDown: externalOnKeyDown,
  ...textareaProps
}) => {
  const [suggestions, setSuggestions] = useState<MentionSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [currentMention, setCurrentMention] = useState<{
    match: string;
    start: number;
    end: number;
    type: MentionType;
  } | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Remove ref from textareaProps if it exists (we need our own ref for positioning)
  const { ref: _, ...restTextareaProps } = textareaProps as any;
  const suggestionsRef = useRef<HTMLDivElement>(null);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Search for mention suggestions
  const searchMentions = async (
    query: string,
    type: MentionType,
    domain?: string,
    institutionName?: string
  ) => {
    if (query.length === 0) {
      setSuggestions([]);
      return;
    }

    try {
      const data = await mentionsApi.searchMentions({
        q: query,
        type,
        limit: 8,
        domain,
        institutionName,
      });

      setSuggestions(data.suggestions || []);
    } catch (error) {
      console.error("Error searching mentions:", error);
      setSuggestions([]);
    }
  };

  // Handle text changes and detect mentions
  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    const cursorPosition = e.target.selectionStart || 0;

    onChange(newValue);

    // Clear existing search timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    // Check if cursor is in a mention
    const mentionAtCursor = getMentionAtPosition(newValue, cursorPosition);

    if (mentionAtCursor && mentionAtCursor.match.startsWith("@")) {
      setCurrentMention(mentionAtCursor);

      const { query, type, domain, institutionName } = extractMentionQuery(
        mentionAtCursor.match
      );

      // Debounced search
      searchTimeoutRef.current = setTimeout(() => {
        searchMentions(query, type, domain, institutionName);
        setShowSuggestions(true);
        setSelectedIndex(0);
      }, 150);
    } else {
      setCurrentMention(null);
      setShowSuggestions(false);
      setSuggestions([]);
    }
  };

  // Handle keyboard navigation in suggestions
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // If suggestions are showing, handle special keys
    if (showSuggestions && suggestions.length > 0) {
      switch (e.key) {
        case "Enter":
          e.preventDefault();
          e.stopPropagation();
          if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
            selectSuggestion(suggestions[selectedIndex]);
          }
          return;

        case "Tab":
          e.preventDefault();
          e.stopPropagation();
          if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
            selectSuggestion(suggestions[selectedIndex]);
          }
          return;

        case "ArrowDown":
          e.preventDefault();
          e.stopPropagation();
          setSelectedIndex((prev) =>
            prev < suggestions.length - 1 ? prev + 1 : 0
          );
          return;

        case "ArrowUp":
          e.preventDefault();
          e.stopPropagation();
          setSelectedIndex((prev) =>
            prev > 0 ? prev - 1 : suggestions.length - 1
          );
          return;

        case "Escape":
          e.preventDefault();
          e.stopPropagation();
          setShowSuggestions(false);
          setSuggestions([]);
          return;
      }
    }

    // For all other cases, call the external handler if it exists
    if (externalOnKeyDown) {
      externalOnKeyDown(e);
    }
  };

  // Select a suggestion and replace the mention
  const selectSuggestion = (suggestion: MentionSuggestion) => {
    if (!currentMention) {
      return;
    }

    const beforeMention = value.slice(0, currentMention.start);
    const afterMention = value.slice(currentMention.end);
    const newValue = beforeMention + suggestion.value + " " + afterMention;

    // Update the value first
    onChange(newValue);

    // Reset state before cursor positioning
    setShowSuggestions(false);
    setSuggestions([]);
    setCurrentMention(null);

    // Set cursor position after the inserted mention (with longer delay to ensure render)
    const newCursorPos = currentMention.start + suggestion.value.length + 1;
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
        textareaRef.current.setSelectionRange(newCursorPos, newCursorPos);
      }
    }, 10); // Increased delay
  };

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Get position for suggestions dropdown
  const getSuggestionPosition = () => {
    if (!textareaRef.current || !currentMention) {
      return { top: 0, left: 0 };
    }

    const textarea = textareaRef.current;
    const style = window.getComputedStyle(textarea);

    // Calculate line height
    const lineHeight =
      parseInt(style.lineHeight) || parseInt(style.fontSize) * 1.5;

    // Count newlines before the mention to determine current line
    const textBeforeMention = value.slice(0, currentMention.start);
    const lineNumber = (textBeforeMention.match(/\n/g) || []).length;

    // Calculate vertical position: account for padding + line number + one line below
    const paddingTop = parseInt(style.paddingTop) || 0;
    const verticalPos = paddingTop + lineNumber * lineHeight + lineHeight;

    // Calculate horizontal position - simple left align for now
    const paddingLeft = parseInt(style.paddingLeft) || 0;

    return {
      top: verticalPos,
      left: paddingLeft,
    };
  };

  const getSuggestionIcon = (suggestion: MentionSuggestion) => {
    switch (suggestion.type) {
      case "user":
        return <User className="h-4 w-4 text-blue-500" />;
      case "bot":
        return <span className="text-base">🤖</span>;
      case "group-relative":
      case "group-absolute":
        return <Users className="h-4 w-4 text-green-500" />;
      default:
        return <Building className="h-4 w-4 text-gray-500" />;
    }
  };

  const position = showSuggestions
    ? getSuggestionPosition()
    : { top: 0, left: 0 };

  return (
    <div className={`relative ${wrapperClassName}`}>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleTextChange}
        onKeyDown={handleKeyDown}
        className={`w-full resize-none ${className}`}
        {...restTextareaProps}
      />

      {showSuggestions && suggestions.length > 0 && (
        <div
          ref={suggestionsRef}
          className="absolute z-50 max-h-48 w-64 overflow-y-auto rounded-lg border border-gray-200 bg-white shadow-lg"
          style={{
            top: position.top,
            left: position.left,
          }}
        >
          {suggestions.map((suggestion, index) => (
            <div
              key={`${suggestion.type}-${suggestion.id}`}
              className={`flex cursor-pointer items-center gap-3 px-3 py-2 transition-colors ${
                index === selectedIndex
                  ? "border-l-2 border-blue-500 bg-blue-50"
                  : "hover:bg-gray-50"
              }`}
              onClick={() => selectSuggestion(suggestion)}
            >
              <div className="flex-shrink-0">
                {getSuggestionIcon(suggestion)}
              </div>
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium text-gray-900">
                  {suggestion.display}
                </div>
                {suggestion.subtitle && (
                  <div className="truncate text-xs text-gray-500">
                    {suggestion.subtitle}
                  </div>
                )}
              </div>
              <div className="font-mono text-xs text-gray-400">
                {suggestion.value}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
