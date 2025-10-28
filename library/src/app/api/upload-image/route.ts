import { NextRequest, NextResponse } from "next/server";
import { getSessionOrRedirect } from "@/lib/auth";
import { BadRequestError, ServiceUnavailableError } from "@/lib/errors";
import { withErrorHandler } from "@/lib/api/error-handler";
import { s3Service } from "@/lib/s3-service";
import { Logger } from "@/lib/logging/logger";

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

  // Check if S3 is configured
  if (!s3Service.isReady()) {
    Logger.warn("S3 not configured, falling back to base64", {
      userId: session.user.id,
    });

    // Fallback to base64 if S3 is not configured
    const bytes = await file.arrayBuffer();
    const base64 = Buffer.from(bytes).toString("base64");
    const dataUrl = `data:${file.type};base64,${base64}`;

    return NextResponse.json({
      imageUrl: dataUrl,
      filename: file.name,
      size: file.size,
      type: file.type,
      storage: "base64",
    });
  }

  // Upload to S3
  try {
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    const result = await s3Service.uploadFile(
      buffer,
      file.type,
      "images",
      file.name
    );

    Logger.info("Image uploaded successfully", {
      userId: session.user.id,
      key: result.key,
      size: result.size,
    });

    return NextResponse.json({
      imageUrl: result.url,
      filename: file.name,
      size: result.size,
      type: file.type,
      storage: "s3",
      key: result.key,
    });
  } catch (error) {
    Logger.error("Failed to upload image to S3", error, {
      userId: session.user.id,
      filename: file.name,
    });

    throw new ServiceUnavailableError(
      "Failed to upload image. Please try again."
    );
  }
});
