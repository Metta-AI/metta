"use client";
import { type FC, useMemo } from "react";
import Select, { type InputActionMeta, type SingleValue } from "react-select";

type Mode = "edit" | "view";

export type EditableSuggestion = {
  text: string;
  hyperlink?: string;
};

type Props = {
  text: string;
  mode: Mode;
  href?: string;
  suggestions?: EditableSuggestion[];
  onChange?: (text: string) => void;
  onModeChange?: (mode: Mode) => void;
};

type OptionType = {
  value: string;
  label: string;
  href: string | undefined;
};

export const EditableTextNode: FC<Props> = ({
  text,
  mode,
  href,
  suggestions,
  onModeChange,
  onChange,
}) => {
  const options = useMemo(() => {
    return suggestions?.map((s) => ({
      value: s.text,
      label: s.text,
      href: s.hyperlink,
    }));
  }, [suggestions]);

  const selectedOption = useMemo(() => {
    return options?.find((option) => option.value === text);
  }, [text, options]);

  function onInputChange(inputValue: string, actionMeta: InputActionMeta) {
    if (actionMeta.action === "input-change") {
      onChange?.(inputValue);
    }
  }

  function onOptionSelect(selectedOption: SingleValue<OptionType>) {
    if (selectedOption) {
      onChange?.(selectedOption.value);

      if (selectedOption.href) {
        window.location.href = selectedOption.href;
      }
    }
  }

  if (mode === "edit") {
    return (
      <span className="inline-block">
        <Select
          autoFocus
          isSearchable
          options={options}
          styles={{
            menu: (base) => ({
              ...base,
              minWidth: "450px",
            }),
            option: (_, props) => ({
              fontSize: "1rem",
              fontWeight: "normal",
              fontFamily: "monospace",
              padding: "3px",
              cursor: "pointer",
              backgroundColor: props.isFocused ? "#e5e7eb" : "white",
            }),
          }}
          value={selectedOption}
          onInputChange={onInputChange}
          onChange={onOptionSelect}
          onBlur={() => onModeChange?.("view")}
          placeholder={text}
        />
      </span>
    );
  }

  if (href) {
    return (
      <a href={href} className="hover:underline">
        {text}
      </a>
    );
  }

  return <span onClick={() => onModeChange?.("edit")}>{text}</span>;
};
