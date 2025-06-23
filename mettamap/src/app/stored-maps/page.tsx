import Link from "next/link";

import { getStoredMapDirs } from "@/lib/api";

import { IndexDirButton } from "./IndexDirButton";

export default async function MapDirsPage() {
  const dirs = await getStoredMapDirs();
  return (
    <div className="flex flex-col gap-2 p-8">
      {dirs.map((dir) => (
        <div key={dir} className="flex items-center gap-1">
          <Link
            className="text-blue-500 hover:underline"
            href={`/stored-maps/dir?dir=${encodeURIComponent(dir)}`}
          >
            {dir}
          </Link>
          <IndexDirButton dir={dir} />
        </div>
      ))}
    </div>
  );
}
