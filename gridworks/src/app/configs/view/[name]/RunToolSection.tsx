"use client";
import { FC } from "react";

import { useCopyTooltip } from "@/hooks/useCopyTooltip";
import { Config } from "@/lib/api";

export const RunToolSection: FC<{ cfg: Config }> = ({ cfg }) => {
  const text = `./tools/run.py ${cfg.maker.path}`;
  const { onClick, floating, render } = useCopyTooltip(text);

  return (
    <div className="mb-8 flex items-center gap-4">
      <h2 className="text-lg font-semibold text-gray-600">Run:</h2>
      <div
        className="my-4 rounded-md border-2 border-gray-300 px-4 py-2 hover:bg-gray-100"
        onClick={onClick}
        style={{ cursor: "pointer" }}
      >
        <code className="text-xs" ref={floating.refs.setReference}>
          <span className="text-gray-500">$ </span>
          {text}
        </code>
        {render()}
      </div>
    </div>
  );
};
