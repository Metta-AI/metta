import { NextRequest, NextResponse } from "next/server";

import { loadInstitution } from "@/posts/data/institutions-server";
import { NotFoundError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ name: string }> }
) {
  try {
    const { name } = await params;
    const institutionName = decodeURIComponent(name);
    const institution = await loadInstitution(institutionName);

    if (!institution) {
      throw new NotFoundError("Institution", institutionName);
    }

    return NextResponse.json(institution);
  } catch (error) {
    return handleApiError(error, { endpoint: "GET /api/institutions/[name]" });
  }
}
