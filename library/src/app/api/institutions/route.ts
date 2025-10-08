import { NextResponse } from "next/server";

import { loadInstitutions } from "@/posts/data/institutions-server";
import { withErrorHandler } from "@/lib/api/error-handler";

export const GET = withErrorHandler(async () => {
  const institutions = await loadInstitutions();
  return NextResponse.json(institutions);
});
