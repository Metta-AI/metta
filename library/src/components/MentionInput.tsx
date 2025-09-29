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
      const params = new URLSearchParams({
        q: query,
        type,
        limit: "8",
      });

      if (domain) {
        params.set("domain", domain);
      }

      if (institutionName) {
        params.set("institutionName", institutionName);
      }

      const response = await fetch(`/api/mentions/search?${params}`);
      const data = await response.json();

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
    // Handle Enter key - check if we're in a mention context
    if (e.key === "Enter") {
      // If suggestions are showing, complete the selected suggestion
      if (showSuggestions && suggestions.length > 0) {
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
          selectSuggestion(suggestions[selectedIndex]);
        }
        return; // Don't call external handler
      }

      // If we're in a mention but suggestions haven't loaded yet, prevent default
      // and wait for suggestions to load
      if (currentMention && currentMention.match.startsWith("@")) {
        e.preventDefault();
        return; // Don't call external handler
      }
    }

    // Handle Tab key for mention completion
    if (e.key === "Tab" && showSuggestions && suggestions.length > 0) {
      e.preventDefault();
      if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
        selectSuggestion(suggestions[selectedIndex]);
      }
      return; // Don't call external handler
    }

    // Handle navigation keys when suggestions are showing
    if (showSuggestions && suggestions.length > 0) {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev < suggestions.length - 1 ? prev + 1 : 0
          );
          return; // Don't call external handler

        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev > 0 ? prev - 1 : suggestions.length - 1
          );
          return; // Don't call external handler

        case "Escape":
          setShowSuggestions(false);
          setSuggestions([]);
          return; // Don't call external handler
      }
    }

    // For all other keys, call the external handler if it exists
    if (externalOnKeyDown) {
      externalOnKeyDown(e);
    }
  };

  // Select a suggestion and replace the mention
  const selectSuggestion = (suggestion: MentionSuggestion) => {
    if (!currentMention || !textareaRef.current) return;

    const beforeMention = value.slice(0, currentMention.start);
    const afterMention = value.slice(currentMention.end);
    const newValue = beforeMention + suggestion.value + " " + afterMention;

    onChange(newValue);

    // Set cursor position after the inserted mention
    const newCursorPos = currentMention.start + suggestion.value.length + 1;
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
        textareaRef.current.setSelectionRange(newCursorPos, newCursorPos);
      }
    }, 0);

    setShowSuggestions(false);
    setSuggestions([]);
    setCurrentMention(null);
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
    const font = `${style.fontSize} ${style.fontFamily}`;

    // Create a temporary span to measure text
    const span = document.createElement("span");
    span.style.font = font;
    span.style.visibility = "hidden";
    span.style.position = "absolute";
    span.textContent = value.slice(0, currentMention.start);
    document.body.appendChild(span);

    const textWidth = span.offsetWidth;
    document.body.removeChild(span);

    return {
      top: textarea.offsetTop + 25, // Rough line height
      left: Math.min(textWidth, textarea.offsetWidth - 200), // Keep suggestions in bounds
    };
  };

  const getSuggestionIcon = (suggestion: MentionSuggestion) => {
    switch (suggestion.type) {
      case "user":
        return <User className="h-4 w-4 text-blue-500" />;
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
        {...textareaProps}
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
