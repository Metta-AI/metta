import { NextRequest, NextResponse } from "next/server";
import { loadInstitution } from "@/posts/data/institutions-server";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ name: string }> }
) {
  try {
    const { name } = await params;
    const institutionName = decodeURIComponent(name);
    const institution = await loadInstitution(institutionName);

    if (!institution) {
      return NextResponse.json(
        { error: "Institution not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(institution);
  } catch (error) {
    console.error("Error loading institution:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
