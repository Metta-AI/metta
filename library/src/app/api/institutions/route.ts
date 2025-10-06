import { NextResponse } from "next/server";

import { loadInstitutions } from "@/posts/data/institutions-server";

export async function GET() {
  try {
    const institutions = await loadInstitutions();
    return NextResponse.json(institutions);
  } catch (error) {
    console.error("Error loading institutions:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
