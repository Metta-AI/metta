import { FC, InputHTMLAttributes } from "react";

type Props = Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "className">;

export const NumberInput: FC<Props> = (props) => {
  return (
    <input
      className="w-full rounded-md border-2 border-gray-300 bg-white px-2 py-0.5"
      {...props}
    />
  );
};
