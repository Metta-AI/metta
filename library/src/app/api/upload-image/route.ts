import { NextRequest, NextResponse } from "next/server";
import { getSessionOrRedirect } from "@/lib/auth";
import { BadRequestError } from "@/lib/errors";
import { withErrorHandler } from "@/lib/api/error-handler";

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/gif",
  "image/webp",
];

export const POST = withErrorHandler(async (request: NextRequest) => {
  // Check authentication
  const session = await getSessionOrRedirect();

  const formData = await request.formData();
  const file = formData.get("image") as File;

  if (!file) {
    throw new BadRequestError("No image file provided");
  }

  // Validate file size
  if (file.size > MAX_FILE_SIZE) {
    throw new BadRequestError("File too large. Maximum size is 10MB.");
  }

  // Validate file type
  if (!ALLOWED_TYPES.includes(file.type)) {
    throw new BadRequestError(
      "Invalid file type. Only JPEG, PNG, GIF, and WebP are allowed."
    );
  }

  // Convert to base64 for now (in production, you'd upload to S3/CDN)
  const bytes = await file.arrayBuffer();
  const base64 = Buffer.from(bytes).toString("base64");
  const dataUrl = `data:${file.type};base64,${base64}`;

  return NextResponse.json({
    imageUrl: dataUrl,
    filename: file.name,
    size: file.size,
    type: file.type,
  });
});
