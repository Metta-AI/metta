import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const OBJECTS_DIR = path.resolve(
  process.cwd(),
  "..",
  "packages",
  "mettagrid",
  "nim",
  "mettascope",
  "data",
  "objects"
);

type RouteParams = {
  params: {
    name: string;
  };
};

export async function GET(_request: Request, { params }: RouteParams) {
  const rawName = params.name;
  const safeName = path.basename(rawName);

  if (!safeName.endsWith(".png")) {
    return NextResponse.json({ error: "Only .png assets are supported" }, { status: 400 });
  }

  const assetPath = path.join(OBJECTS_DIR, safeName);

  try {
    const data = await fs.readFile(assetPath);
    return new NextResponse(data, {
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  } catch (error) {
    console.warn(`Failed to load mettascope asset ${safeName}`, error);
    return NextResponse.json({ error: "Asset not found" }, { status: 404 });
  }
}
