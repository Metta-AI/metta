/**
 * PDF Content Extraction using Claude AI
 *
 * Extracts paper metadata (title, summary, page count) from PDFs using Claude's PDF analysis.
 * Figure extraction has been removed - we only extract text content for AI summaries.
 */

import { generateObject } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { z } from "zod";
import { Logger } from "./logging/logger";

// ===== TYPES =====

export interface ExtractedFigure {
  caption: string;
  pageNumber: number;
  context: string;
  figureNumber: number;
  subpanel?: string;
  confidence: number;
  imageData?: string;
  imageType?: string;
  boundingBox?: { x: number; y: number; width: number; height: number };
  aiDetectedText?: string;
  significance?: string;
  explanation?: string;
}

// ===== CLAUDE SCHEMA =====

const SummarySchema = z.object({
  title: z.string().describe("The paper's title"),
  shortExplanation: z
    .string()
    .describe(
      "A 2-3 sentence high-level explanation of what this paper is about and why it matters"
    ),
  summary: z
    .string()
    .describe(
      "A comprehensive summary of the paper's methodology, findings, and contributions"
    ),
  pageCount: z.number().describe("Total number of pages"),
});

// ===== MAIN FUNCTION =====

/**
 * Extract text content from PDF using Claude AI
 *
 * This function sends the PDF to Claude for analysis and extracts:
 * - Title
 * - Short explanation (2-3 sentences)
 * - Comprehensive summary
 * - Page count
 *
 * Note: Figure extraction has been disabled. The figuresWithImages array will always be empty.
 */
export async function extractPdfContent(pdfBuffer: Buffer): Promise<{
  title: string;
  shortExplanation: string;
  summary: string;
  pageCount: number;
  figuresWithImages: ExtractedFigure[];
}> {
  Logger.info("Starting PDF extraction with Claude AI", {
    bufferSize: pdfBuffer.length,
  });

  try {
    // Validate PDF buffer
    if (pdfBuffer.length === 0) {
      throw new Error("PDF buffer is empty");
    }

    // Check PDF header
    const pdfHeader = pdfBuffer.slice(0, 8).toString();
    if (!pdfHeader.startsWith("%PDF")) {
      throw new Error(`Invalid PDF: header is "${pdfHeader}", expected "%PDF"`);
    }

    Logger.info("Valid PDF detected", {
      version: pdfHeader,
      size: `${Math.round(pdfBuffer.length / 1024)} KB`,
    });

    // Send PDF to Claude for analysis
    const summaryResult = await generateObject({
        model: anthropic("claude-3-5-sonnet-20241022"),
        schema: SummarySchema,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: `Analyze this PDF and provide:

1. **Title**: The paper's exact title
2. **Short Explanation**: 2-3 sentences explaining what this paper is about and why it matters
3. **Summary**: A comprehensive summary of methodology, findings, and contributions
4. **Page Count**: Total number of pages in the document

Focus on extracting accurate information from the paper content.`,
              },
              {
                type: "file",
                data: pdfBuffer,
                mediaType: "application/pdf",
              },
            ],
          },
        ],
        maxRetries: 1,
      });

    Logger.info("Claude analysis complete", {
      title: summaryResult.object.title,
      pageCount: summaryResult.object.pageCount,
      summaryLength: summaryResult.object.summary.length,
    });

    return {
      title: summaryResult.object.title,
      shortExplanation: summaryResult.object.shortExplanation,
      summary: summaryResult.object.summary,
      pageCount: summaryResult.object.pageCount,
      figuresWithImages: [], // Figure extraction disabled
    };
  } catch (error) {
    Logger.error(
      "Error in Claude PDF extraction",
      error instanceof Error ? error : new Error(String(error))
    );
    throw error;
  }
}
