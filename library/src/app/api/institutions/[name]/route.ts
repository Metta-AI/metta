import { NextRequest, NextResponse } from "next/server";

import { withErrorHandler } from "@/lib/api/error-handler";
import { NotFoundError } from "@/lib/errors";
import { loadInstitutionByName } from "@/posts/data/managed-institutions";

/**
 * GET /api/institutions/[name]
 * Get a single institution by name with its papers and authors
 */
export const GET = withErrorHandler(
  async (
    request: NextRequest,
    { params }: { params: Promise<{ name: string }> }
  ) => {
    const { name } = await params;
    const decodedName = decodeURIComponent(name);

    const institution = await loadInstitutionByName(decodedName);

    if (!institution) {
      throw new NotFoundError("Institution", decodedName);
    }

    return NextResponse.json(institution);
  }
);

