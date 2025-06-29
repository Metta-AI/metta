import { FC } from "react";

export const TextInput: FC<{
  name: string;
  placeholder?: string;
}> = ({ name, placeholder }) => {
  return (
    <input
      name={name}
      type="text"
      className="w-full rounded-md border-2 border-gray-300 bg-white px-2 py-1"
      placeholder={placeholder}
    />
  );
};
