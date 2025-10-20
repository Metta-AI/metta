import { PdfReader } from "pdfreader";
import { readFileSync, unlinkSync, writeFileSync } from "fs";
import { v4 as uuidv4 } from "uuid";
import { execSync } from "child_process";

// ===== SYSTEM DEPENDENCY CHECKS =====

/**
 * Check if a system command/tool is available
 */
function checkSystemDependency(command: string, name: string): void {
  try {
    execSync(`${command} --version`, { stdio: ["pipe", "pipe", "pipe"] });
    console.log(`✅ ${name} is available`);
  } catch (error: any) {
    console.error(`❌ ${name} not found in system`);
    console.error(`   Command '${command} --version' failed: ${error.message}`);
    console.error(`   This may cause processing to fail silently`);
    throw new Error(`${name} not available: ${error.message}`);
  }
}
import { generateObject } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { z } from "zod";

// Canvas import with graceful fallback
let createCanvas: any = null;
let loadImage: any = null;
let canvasAvailable = false;

try {
  const canvas = require("canvas");
  createCanvas = canvas.createCanvas;
  loadImage = canvas.loadImage;
  canvasAvailable = true;
} catch (error) {
  canvasAvailable = false;
}

/**
 * PDF text item with coordinates
 */
interface PdfTextItem {
  text: string;
  x: number;
  y: number;
  w: number;
  page: number;
}

/**
 * Figure with extracted image
 */
export interface PdfFigureWithImage {
  caption: string;
  pageNumber: number;
  context: string;
  imageData?: string; // Base64 image data
  imageType?: string; // Image type
  boundingBox?: { x: number; y: number; width: number; height: number };
}

/**
 * LLM analysis schema for figures
 */
const FigureAnalysisSchema = z.object({
  figures: z.array(
    z.object({
      figureId: z.string(), // e.g., "Figure 1a", "Figure 2", etc.
      pageNumber: z.number(),
      description: z.string(),
      layout: z.enum(["single", "horizontal", "vertical", "grid"]),
      coordinates: z.object({
        x: z.number(),
        y: z.number(),
        width: z.number(),
        height: z.number(),
      }),
      confidence: z.enum(["high", "medium", "low"]),
    })
  ),
  reasoning: z.string(),
});

/**
 * Main function: Extract PDF content and figures with images
 */
export async function extractPdfFigures(
  pdfBuffer: Buffer
): Promise<PdfFigureWithImage[]> {
  try {
    console.log("🔍 Starting simplified PDF figure extraction...");

    // Step 1: Extract all text with coordinates
    // Try pdf-parse first for better text quality
    let fullText = "";
    let textItems: PdfTextItem[] = [];

    try {
      console.log("🔄 Trying pdf-parse for better text extraction...");
      const pdfParse = await import("pdf-parse");
      const pdfData = await pdfParse.default(pdfBuffer);
      fullText = pdfData.text;
      console.log("✅ pdf-parse successful");
    } catch (error) {
      console.log("⚠️ pdf-parse failed, falling back to pdfreader...");
      console.log("   Error:", (error as any)?.message);
      textItems = await extractPdfText(pdfBuffer);
      fullText = assembleFullText(textItems);
    }

    // Step 2: LLM analyzes the text and determines all figure locations
    // If we don't have textItems (pdf-parse was used), create dummy page dimensions
    if (textItems.length === 0) {
      // Use standard PDF page dimensions
      textItems = [{ text: "", x: 0, y: 0, w: 612, page: 1 }];
    }
    const llmAnalysis = await analyzeFiguresWithLLM(fullText, textItems);

    if (!llmAnalysis || llmAnalysis.figures.length === 0) {
      console.log("⚠️ No figures found by LLM analysis");
      return [];
    }

    // Step 3: Convert PDF pages to images
    const pageImages = await convertPdfToImages(pdfBuffer);

    // Step 4: Extract each figure image based on LLM coordinates
    const figuresWithImages: PdfFigureWithImage[] = [];

    for (const figure of llmAnalysis.figures) {
      const pageImage = pageImages.find(
        (p) => p.pageNumber === figure.pageNumber
      );
      if (pageImage) {
        const imageData = await extractFigureImage(
          pageImage,
          figure.coordinates
        );

        figuresWithImages.push({
          caption: figure.figureId,
          pageNumber: figure.pageNumber,
          context: figure.description,
          imageData,
          imageType: "image/png",
          boundingBox: figure.coordinates,
        });
      }
    }

    console.log(`✅ Extracted ${figuresWithImages.length} figures with images`);
    return figuresWithImages;
  } catch (error) {
    console.error("❌ Error in PDF figure extraction:", error);
    return [];
  }
}

/**
 * Extract all text items with coordinates from PDF
 */
async function extractPdfText(pdfBuffer: Buffer): Promise<PdfTextItem[]> {
  return new Promise((resolve, reject) => {
    const textItems: PdfTextItem[] = [];
    const allItems: any[] = []; // Debug: collect all items
    let pageCount = 0;
    const reader = new PdfReader();

    reader.parseBuffer(pdfBuffer, (err, item) => {
      if (err) {
        reject(err);
        return;
      }

      if (!item) {
        // Debug: log all item types we found
        const itemTypes = new Set(
          allItems.map((item) => {
            const props = Object.keys(item);
            return `{${props.join(", ")}}`;
          })
        );
        console.log("🔍 PDF item types found:", Array.from(itemTypes));

        // Debug: look for potential image items
        const imageItems = allItems.filter(
          (item) =>
            !item.text &&
            !item.page &&
            (item.x !== undefined || item.y !== undefined)
        );
        console.log(`🖼️ Potential image/graphic items: ${imageItems.length}`);
        if (imageItems.length > 0) {
          console.log("   Sample image items:", imageItems.slice(0, 3));
        }

        resolve(textItems);
        return;
      }

      // Debug: collect all items
      allItems.push(item);

      if ("page" in item) {
        pageCount = Math.max(pageCount, item.page || 0);
        return;
      }

      if ("text" in item && pageCount <= 20) {
        textItems.push({
          text: item.text || "",
          x: (item as any).x || 0,
          y: (item as any).y || 0,
          w: (item as any).w || 0,
          page: pageCount,
        });
      }
    });
  });
}

/**
 * Assemble full text from text items for LLM analysis
 */
function assembleFullText(textItems: PdfTextItem[]): string {
  const sortedItems = textItems.sort((a, b) => {
    if (a.page !== b.page) return a.page - b.page;
    const yDiff = a.y - b.y;
    if (Math.abs(yDiff) < 3) return a.x - b.x; // Same line threshold
    return yDiff;
  });

  const lines: string[] = [];
  let currentLine = "";
  let lastY = -1;
  let lastX = -1;
  let lastWidth = 0;
  let lastPage = -1;

  for (const item of sortedItems) {
    const text = item.text.trim();
    if (!text) continue;

    const isNewPage = item.page !== lastPage;
    const isNewLine = Math.abs(item.y - lastY) > 3; // Increased threshold

    if (isNewPage && currentLine) {
      lines.push(currentLine.trim());
      currentLine = "";
      lines.push(`\n--- PAGE ${item.page} ---\n`);
    } else if (isNewLine && currentLine) {
      lines.push(currentLine.trim());
      currentLine = "";
    }

    // Better spacing logic based on X coordinates
    if (currentLine) {
      const expectedNextX = lastX + lastWidth;
      const actualX = item.x;
      const gap = actualX - expectedNextX;

      // Add space based on the gap between text elements
      if (gap > 3) {
        // Significant gap detected
        if (!currentLine.endsWith(" ")) {
          currentLine += " ";
        }
        // Add extra spaces for large gaps (like between columns)
        if (gap > 30) {
          currentLine += " ";
        }
      }
    }

    currentLine += text;

    lastY = item.y;
    lastX = item.x;
    lastWidth = item.w;
    lastPage = item.page;
  }

  if (currentLine) {
    lines.push(currentLine.trim());
  }

  return lines.join("\n");
}

/**
 * Use LLM to analyze PDF text and determine figure locations
 */
async function analyzeFiguresWithLLM(
  fullText: string,
  textItems: PdfTextItem[]
): Promise<{ figures: any[]; reasoning: string } | null> {
  try {
    console.log("🤖 Analyzing figures with LLM...");

    // Debug: check if we have any figure references in the text
    const figureMatches = fullText.match(/[Ff]igure?\s*\d+[a-z]?/g);
    console.log(
      `🔍 Figure references found in text: ${figureMatches?.length || 0}`
    );
    if (figureMatches) {
      console.log("   References:", figureMatches.slice(0, 5));
    }

    // Debug: show sample of extracted text
    console.log(
      `📄 Sample text (first 500 chars): "${fullText.substring(0, 500)}..."`
    );

    // Debug: log full text for manual analysis
    console.log("=".repeat(80));
    console.log("📄 FULL EXTRACTED TEXT:");
    console.log("=".repeat(80));
    console.log(fullText);
    console.log("=".repeat(80));

    // Calculate page dimensions for coordinate reference
    const pageWidth = Math.max(...textItems.map((item) => item.x + item.w));
    const pageHeight = Math.max(...textItems.map((item) => item.y));

    const prompt = `You are analyzing a PDF's extracted text to identify and locate figures.

TASK: Find all figure references (Figure 1, Figure 2, Figure 1a, Figure 1b, etc.) and determine their precise locations.

PDF TEXT CONTENT:
${fullText.substring(0, 4000)}...

PAGE DIMENSIONS: ${pageWidth} x ${pageHeight} (PDF coordinate space)

INSTRUCTIONS:
1. Find all figure references in the text (Figure 1, Figure 2, Figure 1a, Figure 1b, etc.)
2. For each figure, determine:
   - Exact figure ID (e.g., "Figure 1a", "Figure 2")
   - Which page it appears on
   - Brief description based on surrounding text
   - Layout type: "single" (one image), "horizontal" (side-by-side), "vertical" (stacked), "grid" (2x2 or more)
   - Precise bounding box coordinates where the actual figure image would be (NOT the caption)

COORDINATE GUIDELINES:
- Academic papers typically have figures above their captions
- Figures usually occupy central content area (avoid margins)
- For multi-part figures (1a, 1b): split space appropriately
- Caption text helps identify figure location but you need to estimate where the actual image is

Analyze the text and provide detailed figure locations.`;

    // Debug: log the prompt being sent to LLM
    console.log("=".repeat(80));
    console.log("🤖 LLM PROMPT:");
    console.log("=".repeat(80));
    console.log(prompt);
    console.log("=".repeat(80));

    const result = await generateObject({
      model: anthropic("claude-3-5-sonnet-20241022"),
      temperature: 0.1,
      system:
        "You are an expert at analyzing academic papers and locating figures from text extractions.",
      prompt,
      schema: FigureAnalysisSchema,
    });

    // Debug: log the LLM response
    console.log("=".repeat(80));
    console.log("🤖 LLM RESPONSE:");
    console.log("=".repeat(80));
    console.log(JSON.stringify(result.object, null, 2));
    console.log("=".repeat(80));

    console.log(`✅ LLM found ${result.object.figures.length} figures`);
    result.object.figures.forEach((fig) => {
      console.log(
        `   - ${fig.figureId} on page ${fig.pageNumber} (${fig.confidence} confidence)`
      );
    });

    return result.object;
  } catch (error) {
    console.error("❌ LLM figure analysis failed:", error);
    return null;
  }
}

/**
 * Convert PDF to images using GraphicsMagick (same proven approach as adobe-pdf-extractor.ts)
 */
async function convertPdfToImages(
  pdfBuffer: Buffer
): Promise<Array<{ pageNumber: number; imageBuffer: Buffer }>> {
  const pageImages: Array<{ pageNumber: number; imageBuffer: Buffer }> = [];
  const tempPdfPath = `/tmp/pdf-extract-${uuidv4()}.pdf`;

  try {
    console.log(
      `🔄 Converting PDF (${pdfBuffer.length} bytes) to images using GraphicsMagick...`
    );

    // Check if GraphicsMagick is available
    checkSystemDependency("/usr/bin/gm", "GraphicsMagick");

    // Write PDF buffer to temporary file
    console.log(
      `📝 Writing ${pdfBuffer.length} bytes to temp file: ${tempPdfPath}`
    );
    writeFileSync(tempPdfPath, pdfBuffer);

    // Convert pages (limit to first 10 pages)
    const maxPages = 10;

    for (let pageNum = 1; pageNum <= maxPages; pageNum++) {
      const tempImagePath = `/tmp/pdf-page-${uuidv4()}.png`;

      try {
        // Use GraphicsMagick with proven settings: 400 DPI, 1800x2333, density before input
        const gmCommand = `/usr/bin/gm convert -density 400 '${tempPdfPath}[${pageNum}]' -quality 95 -antialias -resize 1800x2333! -background white -flatten '${tempImagePath}'`;

        try {
          execSync(gmCommand, {
            stdio: ["pipe", "pipe", "pipe"],
            encoding: "utf8",
          });
        } catch (gmError: any) {
          console.error(
            `❌ GraphicsMagick conversion failed for page ${pageNum}`
          );
          console.error(`   Command: ${gmCommand}`);
          console.error(`   Error: ${gmError.message}`);
          console.error(`   Exit code: ${gmError.status}`);
          if (
            gmError.message.includes("command not found") ||
            gmError.message.includes("ENOENT")
          ) {
            console.error(
              "   ⚠️ GraphicsMagick may not be installed in the system"
            );
          }
          throw gmError;
        }

        // Read the generated image
        const imageBuffer = readFileSync(tempImagePath);
        pageImages.push({ pageNumber: pageNum, imageBuffer });

        // Clean up temp image
        unlinkSync(tempImagePath);
      } catch (pageError) {
        // Page doesn't exist or other error, break the loop
        break;
      }
    }

    return pageImages;
  } catch (error) {
    console.error("❌ Error converting PDF to images:", error);
    return [];
  } finally {
    // Clean up temp PDF
    try {
      unlinkSync(tempPdfPath);
    } catch {}
  }
}

/**
 * Extract figure image using coordinates
 */
async function extractFigureImage(
  pageImage: { pageNumber: number; imageBuffer: Buffer },
  coordinates: { x: number; y: number; width: number; height: number }
): Promise<string | undefined> {
  try {
    // Convert PDF coordinates to image coordinates
    const scaleX = 1600 / 612; // PDF width to image width
    const scaleY = 2000 / 792; // PDF height to image height

    const imageCoords = {
      x: Math.round(coordinates.x * scaleX),
      y: Math.round(coordinates.y * scaleY),
      width: Math.round(coordinates.width * scaleX),
      height: Math.round(coordinates.height * scaleY),
    };

    // Crop the image
    const croppedBuffer = await cropImage(pageImage.imageBuffer, imageCoords);

    // Convert to base64
    return croppedBuffer.toString("base64");
  } catch (error) {
    console.error("❌ Error extracting figure image:", error);
    return undefined;
  }
}

/**
 * Crop image using available tools
 */
async function cropImage(
  imageBuffer: Buffer,
  region: { x: number; y: number; width: number; height: number }
): Promise<Buffer> {
  try {
    if (canvasAvailable && createCanvas && loadImage) {
      // Use canvas for cropping
      const img = await loadImage(imageBuffer);
      const canvas = createCanvas(region.width, region.height);
      const ctx = canvas.getContext("2d");

      ctx.drawImage(
        img,
        region.x,
        region.y,
        region.width,
        region.height,
        0,
        0,
        region.width,
        region.height
      );

      return canvas.toBuffer("image/png");
    } else {
      // Fallback to command line tools
      return await cropWithCommandLine(imageBuffer, region);
    }
  } catch (error) {
    console.error("❌ Error cropping image:", error);
    return imageBuffer; // Return original as fallback
  }
}

/**
 * Crop using command line tools
 */
async function cropWithCommandLine(
  imageBuffer: Buffer,
  region: { x: number; y: number; width: number; height: number }
): Promise<Buffer> {
  const tempId = uuidv4();
  const inputPath = `/tmp/crop_input_${tempId}.png`;
  const outputPath = `/tmp/crop_output_${tempId}.png`;

  try {
    writeFileSync(inputPath, imageBuffer);

    const cropGeometry = `${region.width}x${region.height}+${region.x}+${region.y}`;

    try {
      execSync(
        `gm convert "${inputPath}" -crop ${cropGeometry} "${outputPath}"`,
        { stdio: "pipe" }
      );
    } catch {
      execSync(`convert "${inputPath}" -crop ${cropGeometry} "${outputPath}"`, {
        stdio: "pipe",
      });
    }

    const croppedBuffer = readFileSync(outputPath);
    return croppedBuffer;
  } catch (error) {
    console.error("❌ Command line cropping failed:", error);
    return imageBuffer;
  } finally {
    try {
      unlinkSync(inputPath);
    } catch {}
    try {
      unlinkSync(outputPath);
    } catch {}
  }
}
