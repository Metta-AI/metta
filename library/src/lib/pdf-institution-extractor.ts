import { PdfReader } from "pdfreader";
import Anthropic from "@anthropic-ai/sdk";

/**
 * Interface for PDF text items with coordinates
 */
interface PdfTextItem {
  text: string;
  x: number;
  y: number;
  w: number;
}

/**
 * Extract raw text from the first page of a PDF buffer
 */
async function extractFirstPageText(pdfBuffer: Buffer): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const textItems: PdfTextItem[] = [];
    let currentPage = 0;
    let foundFirstPage = false;

    const reader = new PdfReader();

    reader.parseBuffer(pdfBuffer, (err, item) => {
      if (err) {
        reject(err);
        return;
      }

      if (!item) {
        // End of parsing
        // Sort text items by y-coordinate (top to bottom), then x-coordinate (left to right)
        const sortedItems = textItems.sort((a, b) => {
          const yDiff = a.y - b.y;
          if (Math.abs(yDiff) < 1) {
            // If items are on roughly the same line, sort by x-coordinate
            return a.x - b.x;
          }
          return yDiff;
        });

        const textChunks = sortedItems
          .map((item) => item.text.trim())
          .filter((text) => text.length > 0);
        resolve(textChunks);
        return;
      }

      // Handle page metadata
      if ("page" in item) {
        currentPage = item.page || 0;
        if (currentPage > 1) {
          foundFirstPage = true;
        }
        return;
      }

      // Handle text items - only process first page
      if ("text" in item && currentPage <= 1 && !foundFirstPage) {
        textItems.push({
          text: item.text || "",
          x: (item as any).x || 0,
          y: (item as any).y || 0,
          w: (item as any).w || 0,
        });
      }
    });
  });
}

/**
 * Find text chunks that appear near author names
 */
function findTextNearAuthors(
  textChunks: string[],
  authorNames: string[]
): string[] {
  const relevantChunks: string[] = [];

  // Find indices where any part of author names appear
  const authorIndices: number[] = [];

  authorNames.forEach((authorName) => {
    // Split author name into individual words to handle PDF word splitting
    const nameWords = authorName.toLowerCase().split(" ");

    for (let i = 0; i < textChunks.length; i++) {
      const chunk = textChunks[i].toLowerCase();

      // Check if this chunk contains any part of the author name
      if (
        nameWords.some((word) => chunk.includes(word) || word.includes(chunk))
      ) {
        authorIndices.push(i);
      }
    }
  });

  // Remove duplicates and sort
  const uniqueIndices = [...new Set(authorIndices)].sort((a, b) => a - b);

  // Extract text chunks around author name locations
  uniqueIndices.forEach((index) => {
    // Include chunks from a bit before the author name to several chunks after
    const startIndex = Math.max(0, index - 2);
    const endIndex = Math.min(index + 15, textChunks.length);

    for (let i = startIndex; i < endIndex; i++) {
      if (!relevantChunks.includes(textChunks[i])) {
        relevantChunks.push(textChunks[i]);
      }
    }
  });

  return relevantChunks;
}

/**
 * Use LLM to extract institutions from relevant text chunks
 */
async function extractInstitutionsWithLLM(
  textChunks: string[],
  authorNames: string[]
): Promise<string[]> {
  console.log("🤖 Using LLM to extract institutions...");
  console.log("Author names:", authorNames);

  // Get API key from environment
  const anthropicApiKey = process.env.ANTHROPIC_API_KEY;

  if (!anthropicApiKey) {
    console.warn(
      "⚠️ ANTHROPIC_API_KEY not found, falling back to mock parsing"
    );
    return fallbackInstitutionExtraction(textChunks, authorNames);
  }

  try {
    const anthropic = new Anthropic({
      apiKey: anthropicApiKey,
    });

    // Combine text chunks into readable text
    const fullText = textChunks.join(" ");

    // Create a focused prompt for institution extraction
    const prompt = `You are an expert at extracting institutional affiliations from academic paper text.

CONTEXT: This text is from the first page of an academic paper, focusing on the author affiliation section.

AUTHOR NAMES: ${authorNames.join(", ")}

TEXT TO ANALYZE:
${fullText}

TASK: Extract ONLY the institutional affiliations of the paper's authors. Focus on universities, research institutes, companies, or labs that the authors are affiliated with.

STRICT REQUIREMENTS:
1. Return ONLY actual institutional names (universities, institutes, companies, labs)
2. Do NOT include: paper titles, abstract content, author names, email addresses, dates, citation info
3. Each institution should be on its own line
4. Use the most common/official name for each institution
5. If you see fragments like "Google Research" and "Columbia University", list them separately
6. Remove any text that isn't clearly an institution name

EXAMPLES OF GOOD OUTPUT:
Stanford University
Google Research
MIT Computer Science and Artificial Intelligence Laboratory

EXAMPLES OF WHAT TO EXCLUDE:
- "Autoencoding with Sparse Autoencoders" (paper title)
- "July 2025" (dates)
- "John Smith" (author names)
- "smith@university.edu" (email addresses)
- "Abstract" (section headers)

CRITICAL: Return ONLY institution names, one per line. No explanations, no commentary, no additional text. Just the institution names.

If no clear institutions are found, return exactly: NONE

Good response example:
Stanford University
Google Research

Bad response example:
From the text, I can identify:
- Stanford University
- Google Research`;

    const response = await anthropic.messages.create({
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 500,
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });

    const content = response.content[0];
    if (content.type === "text") {
      const institutionText = content.text.trim();

      if (institutionText === "NONE" || !institutionText) {
        console.log("🏛️ LLM found no institutions");
        return [];
      }

      // Parse institutions from LLM response
      const institutions = institutionText
        .split("\n")
        .map((line) => line.trim())
        .filter(
          (line) => line.length > 0 && !line.toLowerCase().includes("none")
        )
        .filter((inst) => {
          // Filter out explanatory text and sentences
          if (
            inst.includes(":") || // Contains colons (explanatory text)
            inst.includes("identify") ||
            inst.includes("found") ||
            inst.includes("text") ||
            inst.includes("given") ||
            inst.includes("clearly") ||
            inst.includes("following") ||
            inst.startsWith("-") || // List markers
            inst.startsWith("•") || // Bullet points
            inst.match(/^\d+\./) // Numbered lists
          ) {
            return false;
          }

          // Basic validation - must be reasonable length and not contain obvious non-institution content
          return (
            inst.length > 3 &&
            inst.length < 100 &&
            !/\d{4}/.test(inst) && // No years
            !/abstract|email|www|http|arxiv/i.test(inst) && // No paper metadata
            !/^independent$/i.test(inst) && // No "Independent"
            !authorNames.some((author) =>
              inst.toLowerCase().includes(author.toLowerCase())
            ) // No author names
          );
        });

      console.log("🏛️ LLM extracted institutions:", institutions);
      return institutions;
    }

    throw new Error("Unexpected response format from LLM");
  } catch (error) {
    console.error("❌ LLM API error:", error);
    console.log("🔄 Falling back to rule-based parsing");
    return fallbackInstitutionExtraction(textChunks, authorNames);
  }
}

/**
 * Fallback institution extraction using rule-based approach
 */
function fallbackInstitutionExtraction(
  textChunks: string[],
  authorNames: string[]
): string[] {
  console.log("🔧 Using fallback rule-based institution extraction");

  const institutionKeywords = [
    "university",
    "institute",
    "college",
    "research",
    "lab",
    "laboratory",
    "center",
    "centre",
    "school",
    "academy",
    "foundation",
    "tech",
    "company",
  ];

  const institutions: string[] = [];
  const fullText = textChunks.join(" ");

  // Look for common patterns
  const patterns = [
    /([A-Z][a-z]+\s+)+University/g,
    /([A-Z][a-z]+\s+)+Institute/g,
    /([A-Z][a-z]+\s+)+(Research|Lab|Laboratory)/g,
    /(Google|Microsoft|Meta|Apple|Amazon)\s+(Research|AI|Labs?)/g,
  ];

  patterns.forEach((pattern) => {
    const matches = fullText.match(pattern);
    if (matches) {
      matches.forEach((match) => {
        const cleaned = match.trim();
        if (cleaned.length > 5 && cleaned.length < 60) {
          institutions.push(cleaned);
        }
      });
    }
  });

  console.log("🏛️ Fallback extracted institutions:", institutions);
  return [...new Set(institutions)]; // Remove duplicates
}

/**
 * Main function to extract institutions from a PDF buffer
 */
export async function extractInstitutionsFromPdf(
  pdfBuffer: Buffer,
  authorNames: string[]
): Promise<string[]> {
  try {
    // Step 1: Extract raw text from first page
    const textChunks = await extractFirstPageText(pdfBuffer);

    if (textChunks.length === 0) {
      console.log("No text found in PDF");
      return [];
    }

    // Step 2: Find text chunks near author names
    const relevantChunks = findTextNearAuthors(textChunks, authorNames);

    if (relevantChunks.length === 0) {
      console.log("No relevant text found near author names");
      return [];
    }

    // Step 3: Use LLM to extract institutions
    const institutions = await extractInstitutionsWithLLM(
      relevantChunks,
      authorNames
    );

    return institutions;
  } catch (error) {
    console.error("Error extracting institutions from PDF:", error);
    return [];
  }
}
