import { MissionMap } from "./MissionMap";

export default async function MissionMapPage({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = await params;

  return <MissionMap name={name} />;
}
