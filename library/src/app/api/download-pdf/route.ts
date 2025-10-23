import { NextRequest, NextResponse } from "next/server";
import { BadRequestError, ServiceUnavailableError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";
import { Logger } from "@/lib/logging/logger";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const url = searchParams.get("url");
    const filename = searchParams.get("filename") || "document.pdf";

    if (!url) {
      throw new BadRequestError("URL parameter is required");
    }

    // Validate URL format (basic security check)
    try {
      new URL(url);
    } catch {
      throw new BadRequestError("Invalid URL format");
    }

    // Fetch the PDF from the external URL
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; LibraryBot/1.0)",
      },
    });

    if (!response.ok) {
      Logger.warn("Failed to fetch external PDF", {
        url,
        status: response.status,
      });
      throw new ServiceUnavailableError(
        `Failed to fetch PDF: ${response.status}`
      );
    }

    const contentType =
      response.headers.get("content-type") || "application/pdf";

    // Ensure it's a PDF
    if (!contentType.includes("pdf") && !contentType.includes("octet-stream")) {
      throw new BadRequestError("URL does not point to a PDF file");
    }

    const pdfBuffer = await response.arrayBuffer();

    // Return the PDF with download headers
    return new NextResponse(pdfBuffer, {
      status: 200,
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": `attachment; filename="${filename}"`,
        "Content-Length": pdfBuffer.byteLength.toString(),
        "Cache-Control": "public, max-age=3600", // Cache for 1 hour
      },
    });
  } catch (error) {
    return handleApiError(error, { endpoint: "GET /api/download-pdf" });
  }
}
