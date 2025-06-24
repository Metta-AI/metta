import clsx from "clsx";
import { FC, useEffect, useState } from "react";

import { loadMettaTileSets } from "@/lib/draw/mettaTileSets";
import { TileSetCollection } from "@/lib/draw/TileSet";

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
    <div className="flex gap-1">
      {Object.keys(tileSets.nameToTileSet).map((key) => {
        const { wrapper, inner } = tileSets.css(key, 32);
        return (
          <button
            key={key}
            onClick={() => setSelectedEntity(key)}
            className={clsx(
              "cursor-pointer",
              selectedEntity === key
                ? "ring-2 ring-blue-500"
                : "hover:ring-2 hover:ring-blue-300"
            )}
            title={key}
          >
            <div style={wrapper}>
              <div style={inner} />
            </div>
          </button>
        );
      })}
    </div>
  );
};
