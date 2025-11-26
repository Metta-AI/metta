import { FC } from "react";

export const FilterInput: FC<{
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}> = ({ value, onChange, placeholder = "Filter..." }) => {
  return (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      className="w-full rounded border border-gray-300 p-2 md:w-1/2 lg:w-1/3"
      onChange={(e) => onChange(e.target.value)}
    />
  );
};
