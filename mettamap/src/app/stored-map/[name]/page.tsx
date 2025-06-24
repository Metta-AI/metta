import { StorableMapViewer } from "@/components/StorableMapViewer";
import { getStoredMap } from "@/lib/api";

export default async function MapPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const name = (await params).name;
  const map = await getStoredMap(name);

  return <StorableMapViewer map={map} />;
}
