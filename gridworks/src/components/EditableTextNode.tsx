"use client";
import { type FC, useEffect, useRef } from "react";

type Mode = "edit" | "view";

export type EditableSuggestion = {
  text: string;
  hyperlink?: string;
};

type Props = {
  text: string;
  mode: Mode;
  withSuggestions?: boolean;
  suggestions?: EditableSuggestion[];
  onChange?: (text: string) => void;
  onModeChange?: (mode: Mode) => void;
};

export const EditableTextNode: FC<Props> = ({
  text,
  mode,
  suggestions,
  onModeChange,
  onChange,
}) => {
  const textNodeRef = useRef<HTMLSpanElement>(null);
  const originalTextRef = useRef<string>(text);
  const blurTimeoutRef = useRef<NodeJS.Timeout>(null);

  useEffect(() => {
    if (mode === "edit" && textNodeRef.current) {
      textNodeRef.current.focus();
      // Move cursor to the end.
      const range = document.createRange();
      range.selectNodeContents(textNodeRef.current);
      range.collapse(false);
      const sel = window.getSelection();
      sel?.removeAllRanges();
      sel?.addRange(range);
    }
  }, [mode]);

  function reset() {
    onModeChange?.("view");
    if (textNodeRef.current) {
      textNodeRef.current.innerText = originalTextRef.current;
      onChange?.(originalTextRef.current);
    }
  }

  function handleBlur() {
    blurTimeoutRef.current = setTimeout(reset, 100);
  }

  function handleSuggestionClick() {
    if (blurTimeoutRef.current) {
      clearTimeout(blurTimeoutRef.current);
      blurTimeoutRef.current = null;
    }
  }

  return (
    <span className="relative inline-block">
      <span
        ref={textNodeRef}
        contentEditable={mode === "edit" ? "plaintext-only" : "false"}
        className="relative"
        suppressContentEditableWarning={true}
        onBlur={handleBlur}
        onInput={(e) => onChange?.((e.target as HTMLElement).innerText)}
      >
        {text}
      </span>

      {mode === "edit" && suggestions && suggestions.length > 0 && (
        <div className="absolute top-10 left-0 z-10 h-64 w-fit overflow-y-scroll border border-gray-300 bg-white p-4 py-1 text-xs text-gray-600 shadow-sm">
          {suggestions && suggestions.length > 0 ? (
            <div className="flex flex-col">
              {suggestions.map((s) => (
                <a
                  key={s.text}
                  title={s.text}
                  className="truncate rounded p-1 hover:bg-gray-200"
                  href={s.hyperlink}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSuggestionClick();
                  }}
                >
                  {s.text}
                </a>
              ))}
            </div>
          ) : (
            <div>No results.</div>
          )}
        </div>
      )}
    </span>
  );
};
