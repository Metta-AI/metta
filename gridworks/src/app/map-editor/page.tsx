import { getMettagridEncoding } from "@/lib/api";

import { MapEditor } from "./MapEditor";

export default async function MapEditorPage() {
  const charToName = await getMettagridEncoding();

  return <MapEditor charToName={charToName} />;
}
