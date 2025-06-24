import clsx from "clsx";
import { FC, useEffect, useRef, useState } from "react";

import { loadMettaTileSets } from "@/lib/draw/mettaTileSets";
import { TileSetCollection } from "@/lib/draw/TileSetCollection";

const ObjectIcon: FC<{
  name: string;
  tileSets: TileSetCollection;
}> = ({ name, tileSets }) => {
  if (name === "empty") {
    return <div className="h-8 w-8 bg-gray-200" />;
  }
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const size = 32;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    const scaledSize = size * dpr;

    canvas.width = scaledSize;
    canvas.height = scaledSize;

    tileSets.draw(name, ctx, 0, 0, scaledSize);
  }, [name, tileSets]);

  return (
    <canvas
      style={{ width: size, height: size }}
      ref={canvasRef}
      className="rounded-sm"
    />
  );
};

const ObjectEntry: FC<{
  name: string;
  onClick: () => void;
  isSelected: boolean;
  tileSets: TileSetCollection;
}> = ({ name, onClick, isSelected, tileSets }) => {
  return (
    <button
      onClick={() => onClick()}
      className={clsx(
        "flex cursor-pointer items-center gap-2",
        isSelected
          ? "bg-blue-100 ring-2 ring-blue-300"
          : "hover:ring-2 hover:ring-blue-300"
      )}
      title={name}
    >
      <ObjectIcon name={name} tileSets={tileSets} />
      <div className="text-sm text-gray-800">{name}</div>
    </button>
  );
};

export const ObjectsPanel: FC<{
  selectedEntity: string;
  setSelectedEntity: (entity: string) => void;
}> = ({ selectedEntity, setSelectedEntity }) => {
  const [tileSets, setTileSets] = useState<TileSetCollection | null>(null);
  useEffect(() => {
    loadMettaTileSets().then(setTileSets);
  }, []);

  if (!tileSets) {
    return null;
  }

  return (
    <div className="flex flex-col gap-1">
      {Object.keys(tileSets.nameToTileSet).map((key) => (
        <ObjectEntry
          key={key}
          name={key}
          onClick={() => setSelectedEntity(key)}
          isSelected={selectedEntity === key}
          tileSets={tileSets}
        />
      ))}
      <ObjectEntry
        name="empty"
        onClick={() => setSelectedEntity("empty")}
        isSelected={selectedEntity === "empty"}
        tileSets={tileSets}
      />
    </div>
  );
};
