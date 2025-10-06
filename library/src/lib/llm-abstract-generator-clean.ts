import { extractPdfWithOpenAI } from "./openai-pdf-extractor";

/**
 * Minimal PDF content structure (for backward compatibility)
 */
interface PdfContent {
  abstract?: string;
  pageCount?: number;
}

/**
 * Structure for LLM-generated enhanced abstracts
 */
export interface LLMAbstract {
  title: string;
  shortExplanation: string;
  summary: string;
  pageCount: number;
  figuresWithImages: EnhancedFigure[];
  pdfUrl?: string;
  homepageUrl?: string;
  generatedAt: Date;
  version: string; // For future schema evolution
}

/**
 * Enhanced figure with OpenAI analysis and extracted image
 */
export interface EnhancedFigure {
  figureNumber: string;
  caption: string;
  explanation: string;
  significance: string;
  pageNumber: number;
  confidence: number;
  imageData?: string; // Base64 image data
  imageType?: string; // Image MIME type
}

/**
 * Generate an enhanced abstract using our new OpenAI + Adobe extraction
 */
export async function generateLLMAbstract(
  paperTitle: string,
  pdfContent?: PdfContent,
  pdfUrl?: string,
  homepageUrl?: string,
  pdfBuffer?: Buffer
): Promise<LLMAbstract> {
  // Use the new enhanced Anthropic extraction if available
  if (process.env.ANTHROPIC_API_KEY) {
    try {
      console.log(
        "🤖 Generating enhanced abstract with OpenAI + Adobe for paper:",
        paperTitle
      );
      return await generateEnhancedAbstractWithOpenAI(
        pdfUrl,
        homepageUrl,
        pdfBuffer
      );
    } catch (error) {
      console.warn(
        "⚠️ Enhanced Anthropic generation failed:",
        error instanceof Error ? error.message : String(error)
      );
    }
  }

  // Fallback to basic structure
  console.log("🔄 Using basic fallback for paper:", paperTitle);
  return {
    title: paperTitle,
    shortExplanation: "Enhanced abstract generation not available.",
    summary: pdfContent?.abstract || "No summary available.",
    pageCount: pdfContent?.pageCount || 0,
    figuresWithImages: [],
    pdfUrl,
    homepageUrl,
    generatedAt: new Date(),
    version: "2.0",
  };
}

/**
 * Generate enhanced abstract using our new OpenAI + Adobe extraction
 */
async function generateEnhancedAbstractWithOpenAI(
  pdfUrl?: string,
  homepageUrl?: string,
  pdfBuffer?: Buffer
): Promise<LLMAbstract> {
  try {
    console.log("🚀 Using enhanced OpenAI + Adobe extraction...");

    // Use provided buffer or fetch from URL if needed
    if (!pdfBuffer && pdfUrl) {
      console.log("📥 Fetching PDF buffer from URL...");
      const response = await fetch(pdfUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch PDF: ${response.status}`);
      }
      pdfBuffer = Buffer.from(await response.arrayBuffer());
    }

    if (!pdfBuffer) {
      throw new Error(
        "PDF buffer not available - no buffer provided and no URL to fetch from"
      );
    }

    // Use our enhanced extraction
    const extraction = await extractPdfWithOpenAI(pdfBuffer);

    // Convert to LLMAbstract format
    const enhancedAbstract: LLMAbstract = {
      title: extraction.title,
      shortExplanation: extraction.shortExplanation,
      summary: extraction.summary,
      pageCount: extraction.pageCount,
      figuresWithImages: extraction.figuresWithImages.map((fig) => ({
        figureNumber: `Figure ${fig.figureNumber}${fig.subpanel ? fig.subpanel : ""}`,
        caption: fig.caption,
        explanation: fig.context,
        significance: fig.aiDetectedText || fig.context,
        pageNumber: fig.pageNumber,
        confidence: fig.confidence,
        imageData: fig.imageData,
        imageType: fig.imageType || "png",
      })),
      pdfUrl,
      homepageUrl,
      generatedAt: new Date(),
      version: "2.0", // Enhanced version with OpenAI + Adobe
    };

    console.log(
      `✅ Enhanced abstract generated with ${enhancedAbstract.figuresWithImages.length} figures`
    );
    return enhancedAbstract;
  } catch (error) {
    console.error("❌ Error in enhanced abstract generation:", error);
    throw error;
  }
}

/**
 * Update LLM abstract if needed (for backward compatibility)
 */
export async function updateLLMAbstractIfNeeded(
  existingAbstract: LLMAbstract,
  paperTitle: string,
  pdfContent?: PdfContent,
  pdfUrl?: string,
  homepageUrl?: string,
  pdfBuffer?: Buffer
): Promise<LLMAbstract | null> {
  // For now, always use the existing abstract if version 2.0
  if (existingAbstract.version === "2.0") {
    return null; // No update needed
  }

  // Regenerate for older versions
  console.log("🔄 Updating abstract to version 2.0...");
  return await generateLLMAbstract(
    paperTitle,
    pdfContent,
    pdfUrl,
    homepageUrl,
    pdfBuffer
  );
}
