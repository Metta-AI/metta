import { ExtendedMapViewer } from "@/components/MapFileViewer";
import { getStoredMap } from "@/server/api";

export default async function MapPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const name = (await params).name;
  const map = await getStoredMap(name);

  return <ExtendedMapViewer mapData={map} />;
}
