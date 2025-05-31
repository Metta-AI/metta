import { ExtendedMapViewer } from "@/components/MapFileViewer";
import { getMap } from "@/server/getMaps";

export default async function MapPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const name = (await params).name;
  const map = await getMap(name);

  return <ExtendedMapViewer mapFile={map} />;
}
