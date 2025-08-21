import { PdfReader } from "pdfreader";
import { extractPdfWithAdobe, PdfFigureWithImage } from "./adobe-pdf-extractor";
import { extractPdfWithOpenAI } from "./openai-pdf-extractor";

/**
 * Interface for PDF text items with coordinates and metadata
 */
interface PdfTextItem {
  text: string;
  x: number;
  y: number;
  w: number;
  page: number;
}

/**
 * Structured PDF content for LLM processing
 */
export interface PdfContent {
  title: string;
  abstract: string;
  introduction: string;
  mainSections: string[];
  figures: PdfFigure[];
  figuresWithImages: PdfFigureWithImage[];
  references: string[];
  fullText: string;
  pageCount: number;
}

/**
 * Information about figures and their captions with spatial data
 */
export interface PdfFigure {
  caption: string;
  pageNumber: number;
  context: string; // Surrounding text for context
  captionBox?: { x: number; y: number; width: number; height: number }; // Caption position
  estimatedFigureBox?: { x: number; y: number; width: number; height: number }; // Estimated figure position
}

/**
 * Extract comprehensive content from a PDF buffer
 */
export async function extractPdfContent(
  pdfBuffer: Buffer
): Promise<PdfContent> {
  const textItems: PdfTextItem[] = [];
  let pageCount = 0;

  return new Promise((resolve, reject) => {
    const reader = new PdfReader();

    reader.parseBuffer(pdfBuffer, async (err, item) => {
      if (err) {
        reject(err);
        return;
      }

      if (!item) {
        // End of parsing - process all collected text
        const content = await processExtractedText(textItems, pageCount);
        resolve(content);
        return;
      }

      // Handle page metadata
      if ("page" in item) {
        pageCount = Math.max(pageCount, item.page || 0);
        return;
      }

      // Handle text items - collect from all pages (limit to first 20 pages for performance)
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
 * Process extracted text items into structured content
 */
async function processExtractedText(
  textItems: PdfTextItem[],
  pageCount: number
): Promise<PdfContent> {
  // Sort text items by page, then by y-coordinate (top to bottom), then x-coordinate (left to right)
  const sortedItems = textItems.sort((a, b) => {
    if (a.page !== b.page) return a.page - b.page;
    const yDiff = a.y - b.y;
    if (Math.abs(yDiff) < 1) {
      return a.x - b.x;
    }
    return yDiff;
  });

  // Combine text items into readable text with some basic structure preservation
  const lines: string[] = [];
  let currentLine = "";
  let lastY = -1;
  let lastPage = -1;

  for (const item of sortedItems) {
    const text = item.text.trim();
    if (!text) continue;

    // Check if we're on a new page or new line
    const isNewPage = item.page !== lastPage;
    const isNewLine = Math.abs(item.y - lastY) > 2; // Threshold for new line

    if (isNewPage && currentLine) {
      lines.push(currentLine.trim());
      currentLine = "";
      lines.push(`\n--- PAGE ${item.page} ---\n`);
    } else if (isNewLine && currentLine) {
      lines.push(currentLine.trim());
      currentLine = "";
    }

    // Add text to current line
    if (currentLine && !currentLine.endsWith(" ") && !text.startsWith(" ")) {
      currentLine += " ";
    }
    currentLine += text;

    lastY = item.y;
    lastPage = item.page;
  }

  // Add final line
  if (currentLine) {
    lines.push(currentLine.trim());
  }

  const fullText = lines.join("\n");

  // Extract structured sections
  const title = extractTitle(lines);
  const abstract = extractAbstract(lines);
  const introduction = extractIntroduction(lines);
  const mainSections = extractMainSections(lines);

  // Basic figure detection (complex extraction handled separately)
  const figures: PdfFigure[] = [];

  const references = extractReferences(lines);

  return {
    title,
    abstract,
    introduction,
    mainSections,
    figures,
    figuresWithImages: [], // Will be populated by enhanced extractor
    references,
    fullText,
    pageCount,
  };
}

/**
 * Extract paper title from text lines
 */
function extractTitle(lines: string[]): string {
  // Look for title in first few lines, typically the largest/boldest text
  for (let i = 0; i < Math.min(10, lines.length); i++) {
    const line = lines[i].trim();

    // Skip page markers and short lines
    if (line.startsWith("---") || line.length < 10) continue;

    // Title is usually one of the first substantial lines
    if (
      line.length > 10 &&
      line.length < 200 &&
      !line.toLowerCase().includes("abstract")
    ) {
      return line;
    }
  }

  return "Unknown Title";
}

/**
 * Extract abstract section from text lines
 */
function extractAbstract(lines: string[]): string {
  const abstractLines: string[] = [];
  let inAbstract = false;

  for (const line of lines) {
    const lowerLine = line.toLowerCase();

    // Start collecting when we see "abstract"
    if (lowerLine.includes("abstract") && !inAbstract) {
      inAbstract = true;
      // If the line contains more than just "abstract", include it
      if (line.trim().length > 10) {
        abstractLines.push(line);
      }
      continue;
    }

    // Stop when we hit introduction or other sections
    if (
      inAbstract &&
      (lowerLine.includes("introduction") ||
        lowerLine.includes("1.") ||
        lowerLine.includes("i.") ||
        lowerLine.includes("keywords"))
    ) {
      break;
    }

    // Collect abstract content
    if (inAbstract && line.trim()) {
      abstractLines.push(line);
    }
  }

  return abstractLines.join(" ").trim();
}

/**
 * Extract introduction section
 */
function extractIntroduction(lines: string[]): string {
  const introLines: string[] = [];
  let inIntro = false;

  for (const line of lines) {
    const lowerLine = line.toLowerCase();

    // Start collecting when we see "introduction"
    if (
      (lowerLine.includes("introduction") || lowerLine.includes("1.")) &&
      !inIntro
    ) {
      inIntro = true;
      if (line.trim().length > 15) {
        introLines.push(line);
      }
      continue;
    }

    // Stop at next major section
    if (
      inIntro &&
      (lowerLine.includes("2.") ||
        lowerLine.includes("method") ||
        lowerLine.includes("related work") ||
        lowerLine.includes("background"))
    ) {
      break;
    }

    // Collect introduction content
    if (inIntro && line.trim()) {
      introLines.push(line);
    }
  }

  return introLines.join(" ").trim();
}

/**
 * Extract main content sections
 */
function extractMainSections(lines: string[]): string[] {
  const sections: string[] = [];
  let currentSection = "";

  for (const line of lines) {
    const trimmed = line.trim();

    // Skip page markers and very short lines
    if (trimmed.startsWith("---") || trimmed.length < 3) continue;

    // Check if this looks like a section header (numbered or short line with keywords)
    const lowerLine = trimmed.toLowerCase();
    const isSectionHeader =
      /^\d+\./.test(trimmed) ||
      (trimmed.length < 50 &&
        (lowerLine.includes("method") ||
          lowerLine.includes("result") ||
          lowerLine.includes("discussion") ||
          lowerLine.includes("conclusion") ||
          lowerLine.includes("experiment") ||
          lowerLine.includes("evaluation")));

    if (isSectionHeader && currentSection) {
      // Save previous section and start new one
      sections.push(currentSection.trim());
      currentSection = trimmed + " ";
    } else {
      // Add to current section
      currentSection += trimmed + " ";
    }
  }

  // Add final section
  if (currentSection.trim()) {
    sections.push(currentSection.trim());
  }

  // Limit to most substantial sections
  return sections.filter((section) => section.length > 100).slice(0, 8); // Limit to 8 main sections
}

/**
 * Extract references section
 */
function extractReferences(lines: string[]): string[] {
  const references: string[] = [];
  let inReferences = false;

  for (const line of lines) {
    const lowerLine = line.toLowerCase();

    // Start collecting when we see "references" or "bibliography"
    if (
      (lowerLine.includes("references") ||
        lowerLine.includes("bibliography")) &&
      !inReferences
    ) {
      inReferences = true;
      continue;
    }

    // Collect reference entries
    if (inReferences && line.trim()) {
      references.push(line.trim());
    }
  }

  return references.slice(0, 50); // Limit to 50 references
}

/**
 * Parse full text from Adobe into structured content sections
 */
function parseFullTextContent(
  fullText: string
): Omit<PdfContent, "figuresWithImages" | "fullText" | "pageCount"> {
  const lines = fullText
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line);

  // Extract title (usually the first substantial line)
  const title =
    lines.find(
      (line) => line.length > 10 && !line.toLowerCase().includes("abstract")
    ) || "Untitled";

  // Extract abstract
  const abstractStart = fullText.toLowerCase().indexOf("abstract");
  const introStart = fullText.toLowerCase().indexOf("introduction");
  let abstract = "";

  if (abstractStart !== -1) {
    const abstractEnd = introStart !== -1 ? introStart : abstractStart + 1000;
    abstract =
      fullText
        .substring(abstractStart, abstractEnd)
        .replace(/^abstract/i, "")
        .trim()
        .substring(0, 500) + "...";
  }

  // Extract introduction
  let introduction = "";
  if (introStart !== -1) {
    introduction =
      fullText
        .substring(introStart, introStart + 800)
        .replace(/^.*?introduction/i, "")
        .trim() + "...";
  }

  // Extract main sections (simplified)
  const sectionPatterns = [
    /\b(methodology|methods|approach)\b/i,
    /\b(results|findings|experiments)\b/i,
    /\b(discussion|analysis)\b/i,
    /\b(conclusion|conclusions)\b/i,
  ];

  const mainSections: string[] = [];
  for (const pattern of sectionPatterns) {
    const match = fullText.match(pattern);
    if (match) {
      const sectionStart = match.index!;
      const sectionText =
        fullText.substring(sectionStart, sectionStart + 300) + "...";
      mainSections.push(sectionText);
    }
  }

  // Extract basic figures info (will be enhanced with actual images)
  const figureMatches = fullText.match(/Figure\s+\d+[^\n.]*/gi) || [];
  const figures: PdfFigure[] = figureMatches.map((match, index) => ({
    caption: match,
    pageNumber: index + 1, // Approximate
    context: match.substring(0, 100),
  }));

  // Extract references (simplified)
  const referencesStart = fullText.toLowerCase().lastIndexOf("references");
  let references: string[] = [];
  if (referencesStart !== -1) {
    const referencesText = fullText.substring(
      referencesStart + 10,
      referencesStart + 500
    );
    references = referencesText
      .split("\n")
      .filter((line) => line.trim().length > 10)
      .slice(0, 5); // First 5 references
  }

  return {
    title,
    abstract,
    introduction,
    mainSections,
    figures,
    references,
  };
}

/**
 * Enhanced PDF content extraction that includes both text and images
 */
export async function extractPdfContentWithImages(
  pdfBuffer: Buffer
): Promise<PdfContent> {
  // Try Anthropic vision first (preferred method)
  if (process.env.ANTHROPIC_API_KEY) {
    try {
      console.log(`🔍 Extracting PDF content with Anthropic Vision...`);

      const openaiResult = await extractPdfWithOpenAI(pdfBuffer);

      // Convert Anthropic figures to expected format
      const figuresWithImages = openaiResult.figuresWithImages.map((fig) => ({
        caption: fig.caption,
        pageNumber: fig.pageNumber,
        context: fig.context,
        imageData: fig.imageData,
        imageType: fig.imageType,
        boundingBox: fig.boundingBox,
      }));

      // Use structured data directly from enhanced Anthropic extraction
      return {
        title: openaiResult.title,
        abstract: openaiResult.shortExplanation, // Use shortExplanation as abstract
        introduction: openaiResult.summary.substring(0, 500) + "...", // Use part of summary as introduction
        mainSections: [openaiResult.summary], // Full summary as main section
        figures: [], // Anthropic figures are already in figuresWithImages
        figuresWithImages: figuresWithImages,
        references: [], // Enhanced extraction doesn't extract references yet
        fullText: openaiResult.summary, // Use summary as fullText to avoid undefined
        pageCount: openaiResult.pageCount,
      };
    } catch (error) {
      console.error(`❌ Error in Anthropic PDF extraction:`, error);
      // console.log(`🔄 Falling back to Adobe extraction...`);
      throw error; // Re-throw to prevent Adobe fallback
    }
  }

  // COMMENTED OUT: Adobe fallback to prevent GraphicsMagick usage
  /*
  // Fallback to Adobe if Anthropic fails or API key not available
  try {
    console.log(`🔍 Extracting PDF content with Adobe PDF Extract...`);

    const adobeResult = await extractPdfWithAdobe(pdfBuffer);

    // Parse the full text to extract structured content (with null check)
    const structuredContent = parseFullTextContent(adobeResult.fullText || "");

    return {
      title: structuredContent.title,
      abstract: structuredContent.abstract,
      introduction: structuredContent.introduction,
      mainSections: structuredContent.mainSections,
      figures: structuredContent.figures,
      figuresWithImages: adobeResult.figuresWithImages,
      references: structuredContent.references,
      fullText: adobeResult.fullText,
      pageCount: adobeResult.pageCount,
    };
  } catch (error) {
    console.error(`❌ Error in Adobe PDF extraction:`, error);
    console.log(`🔄 Falling back to traditional extraction...`);

    // Final fallback to text-only extraction
    return extractPdfContent(pdfBuffer);
  }
  */

  // If we reach here, Anthropic API key is not available
  throw new Error("No Anthropic API key available and Adobe fallback is disabled");
}
