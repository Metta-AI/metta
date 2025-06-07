"use client";
import { FC, PropsWithChildren, useState } from "react";

import { Button } from "./Button";

export const CopyToClipboardButton: FC<PropsWithChildren<{ text: string }>> = ({
  text,
  children,
}) => {
  const [copySuccess, setCopySuccess] = useState<string | null>(null);

  const copyDataToClipboard = () => {
    try {
      navigator.clipboard.writeText(text);
      setCopySuccess("Map data copied to clipboard!");
      setTimeout(() => setCopySuccess(null), 3000);
    } catch {
      setCopySuccess("Failed to copy map data");
      setTimeout(() => setCopySuccess(null), 3000);
    }
  };

  return (
    <div className="flex flex-col items-center justify-start">
      <Button onClick={copyDataToClipboard}>
        {copySuccess ? "Copied!" : children}
      </Button>
    </div>
  );
};
