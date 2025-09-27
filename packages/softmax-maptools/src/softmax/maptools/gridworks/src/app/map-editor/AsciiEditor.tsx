import { FC, useEffect, useState } from "react";
import TextareaAutosize from "react-textarea-autosize";

import { Button } from "@/components/Button";
import { MettaGrid } from "@/lib/MettaGrid";

export const AsciiEditor: FC<{
  grid: MettaGrid;
  setGrid: (grid: MettaGrid) => void;
}> = ({ grid, setGrid }) => {
  const [ascii, setAscii] = useState(grid.toAscii());

  const [state, setState] = useState<
    { type: "idle" } | { type: "applied" } | { type: "error"; error: string }
  >({ type: "idle" });

  useEffect(() => {
    setAscii(grid.toAscii());
  }, [grid]);

  return (
    <div>
      <TextareaAutosize
        value={ascii}
        className="w-full font-mono"
        onChange={(e) => {
          setAscii(e.target.value);
          setState({ type: "idle" });
        }}
      />
      <Button
        onClick={() => {
          try {
            setGrid(MettaGrid.fromAscii(ascii));
            setState({ type: "applied" });
          } catch (e) {
            setState({
              type: "error",
              error: e instanceof Error ? e.message : "Unknown error",
            });
          }
        }}
        theme="primary"
        disabled={ascii === grid.toAscii()}
      >
        Apply
      </Button>
      {state.type === "error" && (
        <div className="text-red-500">{state.error}</div>
      )}
      {state.type === "applied" && (
        <div className="text-green-500">
          Applied, go back to map editor tab to preview
        </div>
      )}
    </div>
  );
};
