import { StorableMapViewer } from "@/components/StorableMapViewer";
import { getStoredMap } from "@/lib/api";

export default async function MapPage({
  searchParams,
}: {
  searchParams: Promise<{ map: string }>;
}) {
  const url = (await searchParams).map;
  const map = await getStoredMap(url);

  return (
    <div className="flex flex-col gap-2 p-8">
      <StorableMapViewer map={map} url={url} />
    </div>
  );
}
