import { NextRequest, NextResponse } from "next/server";
import { extractPdfContent } from "@/lib/pdf-extractor";
import { config } from "@/lib/config";
import { BadRequestError, ServiceUnavailableError } from "@/lib/errors";
import { withErrorHandler } from "@/lib/api/error-handler";
import { Logger } from "@/lib/logging/logger";

export const POST = withErrorHandler(async (request: NextRequest) => {
  // Check for LLM API key
  if (!config.llm.anthropicApiKey) {
    throw new ServiceUnavailableError("LLM API key not configured");
  }

  // Parse the form data
  const formData = await request.formData();
  const file = formData.get("file") as File;

  if (!file) {
    throw new BadRequestError("No file provided");
  }

  // Validate file type
  if (file.type !== "application/pdf") {
    throw new BadRequestError("File must be a PDF");
  }

  // Validate file size (limit to 10MB)
  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    throw new BadRequestError("File size must be less than 10MB");
  }

  Logger.info("Processing PDF for analysis", {
    fileName: file.name,
    fileSize: file.size,
  });

  // Convert file to buffer
  const arrayBuffer = await file.arrayBuffer();
  const pdfBuffer = Buffer.from(arrayBuffer);

  // Extract PDF content
  const startTime = Date.now();
  const result = await extractPdfContent(pdfBuffer);
  const processingTime = (Date.now() - startTime) / 1000;

  Logger.info("PDF analysis complete", {
    fileName: file.name,
    processingTime,
    title: result.title,
    pageCount: result.pageCount,
    summaryLength: result.summary.length,
    figureCount: result.figuresWithImages.length,
  });

  // Return structured results (no file saving)
  return NextResponse.json({
    success: true,
    fileName: file.name,
    fileSize: file.size,
    processingTime: Math.round(processingTime * 10) / 10,
    title: result.title,
    shortExplanation: result.shortExplanation,
    pageCount: result.pageCount,
    textLength: result.summary.length,
    figureCount: result.figuresWithImages.length,
    fullText: result.summary, // Map summary to fullText for backward compatibility
    figures: result.figuresWithImages.map((figure) => ({
      caption: figure.caption,
      pageNumber: figure.pageNumber,
      context: figure.context,
      figureNumber: figure.figureNumber,
      subpanel: figure.subpanel,
      confidence: figure.confidence,
      boundingBox: figure.boundingBox,
      aiDetectedText: figure.aiDetectedText,
      // Include image data for display
      imageData: figure.imageData
        ? `data:image/png;base64,${figure.imageData}`
        : null,
    })),
  });
});

// Handle OPTIONS request for CORS
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}
