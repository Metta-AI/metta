import Link from "next/link";

import { getStoredMapDirs } from "@/server/api";

export default async function MapDirsPage() {
  const dirs = await getStoredMapDirs();
  return (
    <div className="flex flex-col gap-2 p-8">
      {dirs.map((dir) => (
        <Link
          key={dir}
          className="text-blue-500 hover:underline"
          href={`/stored-maps/dir?dir=${encodeURIComponent(dir)}`}
        >
          {dir}
        </Link>
      ))}
    </div>
  );
}
