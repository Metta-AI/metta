import { StyledLink } from "@/components/StyledLink";
import { getStoredMapDirs } from "@/lib/api";
import { viewStoredMapsDirRoute } from "@/lib/routes";

import { IndexDirButton } from "./IndexDirButton";

export default async function MapDirsPage() {
  const dirs = await getStoredMapDirs();
  return (
    <div className="flex flex-col gap-2 p-8">
      {dirs.map((dir) => (
        <div key={dir} className="flex items-center gap-1">
          <StyledLink href={viewStoredMapsDirRoute(dir)}>{dir}</StyledLink>
          <IndexDirButton dir={dir} />
        </div>
      ))}
    </div>
  );
}
