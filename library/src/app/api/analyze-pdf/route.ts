import { NextRequest, NextResponse } from "next/server";
import { extractPdfWithOpenAI } from "@/lib/openai-pdf-extractor";

export async function POST(request: NextRequest) {
  try {
    // Check for Anthropic API key
    if (!process.env.ANTHROPIC_API_KEY) {
      return NextResponse.json(
        { error: "Anthropic API key not configured" },
        { status: 500 }
      );
    }

    // Parse the form data
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    // Validate file type
    if (file.type !== "application/pdf") {
      return NextResponse.json(
        { error: "File must be a PDF" },
        { status: 400 }
      );
    }

    // Validate file size (limit to 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      return NextResponse.json(
        { error: "File size must be less than 10MB" },
        { status: 400 }
      );
    }

    console.log(`🔄 Processing PDF: ${file.name} (${file.size} bytes)`);

    // Convert file to buffer
    const arrayBuffer = await file.arrayBuffer();
    const pdfBuffer = Buffer.from(arrayBuffer);

    // Extract using OpenAI
    const startTime = Date.now();
    const result = await extractPdfWithOpenAI(pdfBuffer);
    const processingTime = (Date.now() - startTime) / 1000;

    console.log(`✅ PDF analysis complete in ${processingTime.toFixed(1)}s`);
    console.log(`   Title: ${result.title}`);
    console.log(`   Pages: ${result.pageCount}`);
    console.log(`   Summary: ${result.summary.length} chars`);
    console.log(`   Figures: ${result.figuresWithImages.length}`);

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
  } catch (error) {
    console.error("❌ PDF analysis error:", error);

    let errorMessage = "Internal server error";
    if (error instanceof Error) {
      errorMessage = error.message;

      // Provide more specific error messages for common issues
      if (error.message.includes("API")) {
        errorMessage =
          "Anthropic API error. Please check your API key and credits.";
      } else if (error.message.includes("GraphicsMagick")) {
        errorMessage =
          "PDF conversion error. GraphicsMagick may not be installed.";
      } else if (error.message.includes("timeout")) {
        errorMessage =
          "Processing timeout. The PDF may be too complex or large.";
      }
    }

    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

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
