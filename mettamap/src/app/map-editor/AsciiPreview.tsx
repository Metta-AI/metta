import { FC } from "react";
import TextareaAutosize from "react-textarea-autosize";

export const AsciiPreview: FC<{ ascii: string }> = ({ ascii }) => {
  return (
    <TextareaAutosize readOnly value={ascii} className="w-full font-mono" />
  );
};
