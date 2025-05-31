import { FC } from "react";

import { loadMapIndex } from "@/server/loadMapIndex";

import { InnerMapFilter } from "./InnerMapFilter";

export const MapFilter: FC = async () => {
  const mapIndex = await loadMapIndex();

  return (
    <div className="border-b bg-white">
      <div className="px-8 py-4">
        <InnerMapFilter mapIndex={mapIndex} />
      </div>
    </div>
  );
};
