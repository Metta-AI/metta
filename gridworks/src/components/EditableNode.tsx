"use client";
import { useRouter } from "next/navigation";
import { type FC, useMemo, useState } from "react";
import Select, { type InputActionMeta, type SingleValue } from "react-select";

import { EditIcon } from "@/components/icons/EditIcon";

type Mode = "edit" | "view";

export type EditableSuggestion = {
  text: string;
  href: string;
};

type Props = {
  children: React.ReactNode;
  text: string;
  suggestions?: EditableSuggestion[];
  onChange?: (text: string) => void;
};

type OptionType = {
  value: string;
  label: string;
  href: string;
};

export const EditableNode: FC<Props> = ({
  children,
  text,
  suggestions,
  onChange,
}) => {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("view");

  const options = useMemo(() => {
    return suggestions?.map((s) => ({
      value: s.text,
      label: s.text,
      href: s.href,
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
      router.push(selectedOption.href);
    }
  }

  return (
    <span className="inline-flex items-center gap-1">
      {mode === "edit" ? (
        <Select
          autoFocus
          isSearchable
          options={options}
          styles={{
            control: (base) => ({
              ...base,
              minWidth: "200px",
            }),
            menu: (base) => ({
              ...base,
              minWidth: "fit-content",
            }),
            menuList: (base) => ({
              ...base,
              minWidth: "fit-content",
            }),
            option: (_, props) => ({
              fontSize: "1rem",
              fontWeight: "normal",
              fontFamily: "monospace",
              padding: "3px",
              cursor: "pointer",
              backgroundColor: props.isFocused ? "#e5e7eb" : "white",
            }),
            noOptionsMessage: (base) => ({
              ...base,
              fontSize: "1rem",
              fontWeight: "normal",
              fontFamily: "monospace",
              padding: "3px",
            }),
          }}
          value={selectedOption}
          onInputChange={onInputChange}
          onChange={onOptionSelect}
          onBlur={() => {
            setMode("view");
            onChange?.("");
          }}
          placeholder={text}
        />
      ) : (
        <span>{children}</span>
      )}
      <EditIcon
        className="cursor-pointer"
        onClick={() => setMode((m) => (m === "edit" ? "view" : "edit"))}
      />
    </span>
  );
};
