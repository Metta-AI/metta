import { type SearchParams } from "nuqs/server";
import { FC, ReactNode, Suspense } from "react";

import { MapLoader } from "@/components/MapLoader";
import { findStoredMaps } from "@/lib/api";

import { LoadMoreButton } from "./LoadMoreButton";
import { MapFilter } from "./MapFilter";
import { paramsLoader } from "./params";

const MapListShell: FC<{
  total: string | number;
  children?: ReactNode;
  hasMore?: boolean;
}> = async ({ total, hasMore = false, children = null }) => {
  return (
    <div className="p-8">
      <div className="mb-4 text-sm text-gray-600">
        Total maps: <span className="font-semibold">{total}</span>
      </div>

      <div className="flex flex-col gap-4">{children}</div>
      {hasMore && (
        <div className="mt-8">
          <LoadMoreButton />
        </div>
      )}
    </div>
  );
};

const LoadMapsList: FC<{ searchParams: Promise<SearchParams> }> = async ({
  searchParams,
}) => {
  const { dir, filter, limit } = await paramsLoader(searchParams);

  if (!dir) {
    return <div>No directory selected</div>;
  }

  const mapsMetadata = await findStoredMaps(dir, filter ?? undefined);

  const hasMore = mapsMetadata.length > limit;
  return (
    <MapListShell total={mapsMetadata.length} hasMore={hasMore}>
      {mapsMetadata.slice(0, limit).map((map) => (
        <div key={map.url} className="rounded-md border-2 border-gray-300 p-4">
          <MapLoader mapUrl={map.url} />
        </div>
      ))}
    </MapListShell>
  );
};

export default async function MapsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  let filterKey = await searchParams.then((params) => params.filter);
  if (Array.isArray(filterKey)) {
    filterKey = filterKey.join(",");
  }

  return (
    <div>
      <div className="border-b bg-white">
        <div className="px-8 py-4">
          <MapFilter />
        </div>
      </div>
      <Suspense
        key={filterKey}
        fallback={<MapListShell total="..." hasMore={false} />}
      >
        <LoadMapsList searchParams={searchParams} />
      </Suspense>
    </div>
  );
}
