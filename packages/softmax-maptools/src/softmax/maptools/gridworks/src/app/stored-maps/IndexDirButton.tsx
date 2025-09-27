"use client";
import { FC } from "react";

import { Button } from "@/components/Button";
import { indexDir } from "@/lib/api";

export const IndexDirButton: FC<{ dir: string }> = ({ dir }) => {
  return (
    <Button size="sm" theme="primary" onClick={() => indexDir(dir)}>
      Index
    </Button>
  );
};
