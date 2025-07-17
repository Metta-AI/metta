import { FC, useId } from "react";

export const TextInput: FC<{
  name: string;
  label: string;
}> = ({ name, label }) => {
  const id = useId();

  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-bold text-gray-700">
        {label}
      </label>
      <input
        type="text"
        id={id}
        name={name}
        className="rounded-md border border-gray-300 px-2 py-1"
      />
    </div>
  );
};
