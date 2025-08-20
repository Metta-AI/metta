// ✅ NEW: Hybrid OpenAI + Adobe PDF extraction following batch script approach EXACTLY
// 📝 Step 1: OpenAI identifies key figures + provides summary (generateObject + zod)
// 🔧 Step 2: Get raw Adobe elements (same as debug-adobe-raw.ts)
// 🎯 Step 3: Create semantic mappings (same as batch script)
// 🔍 Step 4: Extract ONLY semantically validated figures (same as batch script)
// 💾 Step 5: Save extracted figures with correct coordinates
// 🏆 Result: Same proven approach as working batch script

import { generateObject } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import {
  writeFileSync,
  mkdirSync,
  existsSync,
  statSync,
  readdirSync,
  createReadStream,
  unlinkSync,
} from "fs";
import path from "path";
import { execSync } from "child_process";

// ===== EXACT BATCH SCRIPT INTERFACES =====

interface AdobeElement {
  ObjectID: number;
  Page: number;
  Path?: string;
  Text?: string;
  Bounds?: number[];
}

interface SemanticMapping {
  objectID: number;
  semanticLabel: string;
  figureNumber: number;
  subpanel?: string;
  confidence: number;
}

interface OpenAIPdfFigure {
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
}

// ===== OPENAI SCHEMA =====

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
  keyFigures: z
    .array(
      z.object({
        figureNumber: z
          .string()
          .describe(
            "EXACT figure number/identifier including sub-panels (e.g., 'Figure 1', 'Figure 2a', 'Figure 2b', 'Figure 3c')"
          ),
        caption: z.string().describe("The figure's caption or title"),
        significance: z
          .string()
          .describe("Why this figure is important to the paper"),
        explanation: z
          .string()
          .describe(
            "A detailed explanation of what this figure shows and how to interpret it"
          ),
        pageNumber: z
          .number()
          .optional()
          .describe("Page number if identifiable"),
      })
    )
    .describe(
      "3-5 most important figures with precise identifiers and detailed explanations"
    )
    .optional(),
});

// ===== EXACT BATCH SCRIPT FUNCTIONS (COPIED DIRECTLY) =====

function analyzeCaption(captionText: string): {
  figureNumber: number;
  expectedSubpanels: string[];
  isSingleFigure: boolean;
} | null {
  // EXACT COPY FROM WORKING BATCH SCRIPT
  const figureMatch = captionText.match(/^Figure (\d+):/);
  if (!figureMatch) return null;

  const figureNumber = parseInt(figureMatch[1]);
  const foundSubpanels = new Set<string>();

  // Extract individual subpanels like (a), (b), (c), (a):, (b):
  const individualMatches = captionText.matchAll(/\(([a-z])\):?/g);
  for (const match of individualMatches) {
    foundSubpanels.add(match[1]);
  }

  // Extract ranges like (a-d) or (a–d)
  const rangeMatches = captionText.matchAll(/\(([a-z])[-–]([a-z])\)/g);
  for (const match of rangeMatches) {
    const start = match[1].charCodeAt(0);
    const end = match[2].charCodeAt(0);
    for (let i = start; i <= end; i++) {
      foundSubpanels.add(String.fromCharCode(i));
    }
  }

  const expectedSubpanels = Array.from(foundSubpanels).sort();
  const isSingleFigure = expectedSubpanels.length === 0;

  return {
    figureNumber,
    expectedSubpanels,
    isSingleFigure,
  };
}

function findFigureGroup(
  allElements: AdobeElement[],
  figureNumber: number,
  page: number
): AdobeElement[] {
  return allElements.filter((element) => {
    // Must be on the same page
    if (element.Page !== page) return false;

    // Must be a figure path (not text/captions)
    if (!element.Path?.includes("Figure")) return false;

    // Must not have text (actual figure, not caption)
    if (element.Text) return false;

    return true;
  });
}

function isActualFigure(element: AdobeElement): boolean {
  // Must have figure path
  if (!element.Path?.includes("Figure")) return false;

  // Must not be text/caption
  if (element.Text) return false;

  // Must have valid bounds
  if (!element.Bounds || element.Bounds.length !== 4) return false;

  // Check minimum size (filter out tiny elements)
  const [x1, y1, x2, y2] = element.Bounds;
  const width = Math.abs(x2 - x1);
  const height = Math.abs(y2 - y1);

  return width > 20 && height > 20;
}

// EXACT COPY FROM WORKING BATCH SCRIPT
function detectStructuralMultiPanel(
  figureGroup: AdobeElement[],
  actualFigures: AdobeElement[]
): boolean {
  if (actualFigures.length <= 1) return false;

  // Look for subpanel labels like (a), (b), (c) in the figure group
  let subpanelLabelCount = 0;
  for (const element of figureGroup) {
    if (element.Text && /^\([a-z]\)/.test(element.Text.trim())) {
      subpanelLabelCount++;
    }
  }

  // If we have multiple figures AND subpanel labels, it's multi-panel
  return subpanelLabelCount >= 2 && actualFigures.length >= 2;
}

function createStructuralSubpanelMappings(
  elements: AdobeElement[],
  figureNumber: number,
  subpanels: string[]
): SemanticMapping[] {
  const actualFigures = elements.filter(isActualFigure);

  // Sort by position for consistent mapping
  const sortedFigures = actualFigures.sort((a, b) => {
    const aY = a.Bounds ? a.Bounds[1] : 0;
    const bY = b.Bounds ? b.Bounds[1] : 0;
    if (Math.abs(aY - bY) > 10) return bY - aY;
    const aX = a.Bounds ? a.Bounds[0] : 0;
    const bX = b.Bounds ? b.Bounds[0] : 0;
    return aX - bX;
  });

  return sortedFigures.map((element, index) => ({
    objectID: element.ObjectID,
    semanticLabel: `Figure ${figureNumber}${subpanels[index] || ""}`,
    figureNumber,
    subpanel: subpanels[index],
    confidence: 0.95,
  }));
}

function createSemanticMappings(
  allElements: AdobeElement[]
): Map<number, SemanticMapping> {
  const mappings = new Map<number, SemanticMapping>();

  // Find all caption elements
  const captionElements = allElements
    .filter((el) => el.Text && el.Text.trim())
    .sort((a, b) => {
      if (a.Page !== b.Page) return a.Page - b.Page;
      const aY = a.Bounds ? a.Bounds[1] : 0;
      const bY = b.Bounds ? b.Bounds[1] : 0;
      if (Math.abs(aY - bY) > 5) return bY - aY;
      const aX = a.Bounds ? a.Bounds[0] : 0;
      const bX = b.Bounds ? b.Bounds[0] : 0;
      return aX - bX;
    });

  console.log(
    `📋 Found ${captionElements.filter((el) => el.Text?.match(/^Figure\s+\d+/i)).length} caption anchors for semantic mapping`
  );

  // Process each caption
  for (const caption of captionElements) {
    if (!caption.Text) continue;

    // Debug: Log all figure-related captions
    if (caption.Text.toLowerCase().includes("figure")) {
      console.log(`🔍 Caption found: "${caption.Text.substring(0, 100)}..."`);
    }

    // EXACT COPY FROM WORKING BATCH SCRIPT - TWO-PHASE DETECTION
    const analysis = analyzeCaption(caption.Text);
    if (!analysis) continue;

    const figureGroup = findFigureGroup(
      allElements,
      analysis.figureNumber,
      caption.Page
    );
    const actualFigures = figureGroup.filter(isActualFigure);

    // Determine if this is actually multi-panel by structural analysis
    const isStructuralMultiPanel = detectStructuralMultiPanel(
      figureGroup,
      actualFigures
    );
    const isSingleFigure = analysis.isSingleFigure && !isStructuralMultiPanel;

    console.log(
      `   📍 Figure ${analysis.figureNumber}: ${isSingleFigure ? "Single" : "Multi-panel"} (${actualFigures.length} figures) ${isStructuralMultiPanel ? "[structural detection]" : "[caption detection]"}`
    );

    if (isSingleFigure) {
      // Single figure
      if (actualFigures.length > 0) {
        const mainFigure = actualFigures.reduce((largest, current) => {
          const largestArea =
            (largest.Bounds![2] - largest.Bounds![0]) *
            (largest.Bounds![3] - largest.Bounds![1]);
          const currentArea =
            (current.Bounds![2] - current.Bounds![0]) *
            (current.Bounds![3] - current.Bounds![1]);
          return currentArea > largestArea ? current : largest;
        });

        mappings.set(mainFigure.ObjectID, {
          objectID: mainFigure.ObjectID,
          semanticLabel: `Figure ${analysis.figureNumber}`,
          figureNumber: analysis.figureNumber,
          confidence: 0.9,
        });
      }
    } else {
      // Multi-panel figure - create subpanel mappings
      const sortedFigures = actualFigures.sort((a, b) => {
        const aY = a.Bounds ? a.Bounds[1] : 0;
        const bY = b.Bounds ? b.Bounds[1] : 0;
        if (Math.abs(aY - bY) > 10) return bY - aY;
        const aX = a.Bounds ? a.Bounds[0] : 0;
        const bX = b.Bounds ? b.Bounds[0] : 0;
        return aX - bX;
      });

      sortedFigures.forEach((element, index) => {
        const subpanel = String.fromCharCode(97 + index); // a, b, c, d...
        mappings.set(element.ObjectID, {
          objectID: element.ObjectID,
          semanticLabel: `Figure ${analysis.figureNumber}${subpanel}`,
          figureNumber: analysis.figureNumber,
          subpanel,
          confidence: 0.95,
        });
      });
    }
  }

  return mappings;
}

function filterActualFigures(allElements: AdobeElement[]): Set<number> {
  return new Set(
    allElements
      .filter((element) => isActualFigure(element))
      .map((element) => element.ObjectID)
  );
}

// ===== RAW ADOBE DATA EXTRACTION (SAME AS debug-adobe-raw.ts) =====

async function getRawAdobeElements(pdfBuffer: Buffer): Promise<AdobeElement[]> {
  console.log("🔍 Calling Adobe PDF Extract API for raw elements...");

  const {
    PDFServices,
    ServicePrincipalCredentials,
    ExtractPDFParams,
    ExtractElementType,
    ExtractPDFJob,
    ExtractPDFResult,
    MimeType,
  } = await import("@adobe/pdfservices-node-sdk");
  const { v4: uuidv4 } = await import("uuid");

  const tempFileName = `temp-pdf-${uuidv4()}.pdf`;

  try {
    const credentials = new ServicePrincipalCredentials({
      clientId: process.env.ADOBE_CLIENT_ID!,
      clientSecret: process.env.ADOBE_CLIENT_SECRET!,
    });

    const pdfServices = new PDFServices({ credentials });
    writeFileSync(tempFileName, pdfBuffer);

    const inputAsset = await pdfServices.upload({
      readStream: createReadStream(tempFileName),
      mimeType: MimeType.PDF,
    });

    const params = new ExtractPDFParams({
      elementsToExtract: [ExtractElementType.TEXT, ExtractElementType.TABLES],
    });

    const job = new ExtractPDFJob({ inputAsset, params });
    const pollingURL = await pdfServices.submit({ job });
    const pdfServicesResponse = await pdfServices.getJobResult({
      pollingURL,
      resultType: ExtractPDFResult,
    });

    console.log("💾 Downloading extraction results...");
    const resultAsset = pdfServicesResponse.result?.resource;
    if (!resultAsset) {
      throw new Error("No result asset found in Adobe response");
    }
    const streamAsset = await pdfServices.getContent({ asset: resultAsset });

    // Save the zip file temporarily using streaming (same as debug-adobe-raw.ts)
    const tempZipPath = `/tmp/adobe-extract-${uuidv4()}.zip`;
    const outputStream = require("fs").createWriteStream(tempZipPath);
    streamAsset.readStream.pipe(outputStream);

    // Wait for streaming to complete
    await new Promise((resolve) => {
      outputStream.on("finish", resolve);
    });

    // Extract and parse the JSON result
    const AdmZip = require("adm-zip");
    const zip = new AdmZip(tempZipPath);
    const jsonEntry = zip
      .getEntries()
      .find((entry: any) => entry.entryName === "structuredData.json");

    if (!jsonEntry) {
      throw new Error("Could not find structuredData.json in Adobe response");
    }

    const extractionResult = JSON.parse(jsonEntry.getData().toString("utf8"));

    // Cleanup
    unlinkSync(tempZipPath);

    console.log(
      `✅ Got ${extractionResult.elements?.length || 0} raw Adobe elements`
    );
    return extractionResult.elements || [];
  } finally {
    if (existsSync(tempFileName)) {
      unlinkSync(tempFileName);
    }
  }
}

// ===== FIGURE EXTRACTION (SAME AS BATCH SCRIPT) =====

async function extractFigureFromElement(
  element: AdobeElement,
  semanticLabel: string,
  pdfBuffer: Buffer
): Promise<{ imageData: string; imageType: string } | null> {
  if (!element.Bounds || element.Bounds.length !== 4) {
    console.log(`⚠️ Invalid bounds for ${semanticLabel}`);
    return null;
  }

  try {
    // Convert PDF page to image (same logic as batch script)
    const pageNum = element.Page;
    const tempPdfPath = `/tmp/pdf-page-${Math.random().toString(36).substring(7)}.pdf`;
    const tempImagePath = `/tmp/pdf-page-${Math.random().toString(36).substring(7)}.png`;

    writeFileSync(tempPdfPath, pdfBuffer);

    // Convert page to image with exact same settings as batch script
    execSync(
      `/usr/bin/gm convert -density 400 '${tempPdfPath}[${pageNum}]' -quality 95 -antialias -resize 1800x2333! -background white -flatten '${tempImagePath}'`,
      { stdio: "pipe" }
    );

    if (!existsSync(tempImagePath)) {
      console.log(`❌ Failed to convert page ${pageNum} for ${semanticLabel}`);
      return null;
    }

    // Crop figure using exact same coordinate transformation
    const [x1, y1, x2, y2] = element.Bounds;
    const pdfWidth = 612;
    const pdfHeight = 792;
    const imageWidth = 1800;
    const imageHeight = 2333;

    const scaleX = imageWidth / pdfWidth;
    const scaleY = imageHeight / pdfHeight;

    const x = Math.round(x1 * scaleX);
    const y = Math.round((pdfHeight - y2) * scaleY);
    const width = Math.round((x2 - x1) * scaleX);
    const height = Math.round((y2 - y1) * scaleY);

    const outputPath = `/tmp/cropped-${Math.random().toString(36).substring(7)}.png`;

    execSync(
      `/usr/bin/gm convert '${tempImagePath}' -crop ${width}x${height}+${x}+${y} '${outputPath}'`,
      { stdio: "pipe" }
    );

    if (existsSync(outputPath)) {
      const imageBuffer = require("fs").readFileSync(outputPath);
      const imageData = imageBuffer.toString("base64");

      // Cleanup
      [tempPdfPath, tempImagePath, outputPath].forEach((file) => {
        if (existsSync(file)) unlinkSync(file);
      });

      return { imageData, imageType: "png" };
    }

    return null;
  } catch (error) {
    console.log(`❌ Error extracting ${semanticLabel}:`, error);
    return null;
  }
}

// ===== MAIN EXTRACTION FUNCTION =====

async function extractFiguresWithSemanticValidation(
  keyFigures: any[],
  rawAdobeElements: AdobeElement[],
  semanticMappings: Map<number, SemanticMapping>,
  pdfBuffer: Buffer
): Promise<OpenAIPdfFigure[]> {
  const matchedFigures: OpenAIPdfFigure[] = [];
  const outputDir = path.resolve(process.cwd(), "identified-figures");

  mkdirSync(outputDir, { recursive: true });

  console.log(
    `🎯 Processing ${keyFigures.length} OpenAI figures against ${semanticMappings.size} semantic mappings...`
  );

  for (const keyFig of keyFigures) {
    console.log(`🔍 Looking for: ${keyFig.figureNumber}`);

    let bestMatch: {
      mapping: SemanticMapping;
      element: AdobeElement;
      confidence: number;
    } | null = null;

    // Try exact semantic label match
    for (const [objectID, mapping] of semanticMappings) {
      if (mapping.semanticLabel === keyFig.figureNumber) {
        const element = rawAdobeElements.find((el) => el.ObjectID === objectID);
        if (element) {
          bestMatch = { mapping, element, confidence: 0.98 };
          console.log(`✅ EXACT semantic match: ${mapping.semanticLabel}`);
          break;
        }
      }
    }

    // Smart fallback: If OpenAI said "Figure 1" but we have "Figure 1a", map to first subpanel
    if (!bestMatch && keyFig.figureNumber.match(/^Figure \d+$/)) {
      const figureNumber = keyFig.figureNumber.replace("Figure ", "");
      const firstSubpanel = `Figure ${figureNumber}a`;

      for (const [objectID, mapping] of semanticMappings) {
        if (mapping.semanticLabel === firstSubpanel) {
          const element = rawAdobeElements.find(
            (el) => el.ObjectID === objectID
          );
          if (element) {
            bestMatch = { mapping, element, confidence: 0.95 };
            console.log(
              `🎯 Smart subpanel mapping: "${keyFig.figureNumber}" → "${firstSubpanel}"`
            );
            break;
          }
        }
      }
    }

    // Try fuzzy figure number matching
    if (!bestMatch) {
      const keyFigNumber = parseInt(
        keyFig.figureNumber.match(/\d+/)?.[0] || "0"
      );
      const keyFigPanel = keyFig.figureNumber
        .match(/[a-z]$/i)?.[0]
        ?.toLowerCase();

      for (const [objectID, mapping] of semanticMappings) {
        if (mapping.figureNumber === keyFigNumber) {
          const panelMatch = keyFigPanel
            ? mapping.subpanel === keyFigPanel
            : !mapping.subpanel;

          if (panelMatch) {
            const element = rawAdobeElements.find(
              (el) => el.ObjectID === objectID
            );
            if (element) {
              const confidence = keyFigPanel ? 0.95 : 0.85;
              if (!bestMatch || confidence > bestMatch.confidence) {
                bestMatch = { mapping, element, confidence };
                console.log(
                  `✅ Fuzzy semantic match: ${mapping.semanticLabel} (confidence: ${confidence})`
                );
              }
            }
          }
        }
      }
    }

    if (bestMatch) {
      const { mapping, element } = bestMatch;

      // Extract the figure image (same as batch script)
      const extraction = await extractFigureFromElement(
        element,
        mapping.semanticLabel,
        pdfBuffer
      );

      if (extraction) {
        // Save the image
        const filename = `${keyFig.figureNumber.replace(/[^a-zA-Z0-9]/g, "_")}.png`;
        const filepath = path.join(outputDir, filename);
        const imageBuffer = Buffer.from(extraction.imageData, "base64");
        writeFileSync(filepath, imageBuffer);

        if (existsSync(filepath)) {
          const stats = statSync(filepath);
          console.log(`✅ Saved: ${filename} (${stats.size} bytes)`);
        }

        matchedFigures.push({
          caption: keyFig.caption,
          pageNumber: element.Page + 1, // Adobe uses 0-based, we want 1-based
          context: keyFig.explanation || keyFig.significance,
          figureNumber: mapping.figureNumber,
          subpanel: mapping.subpanel,
          confidence: bestMatch.confidence,
          imageData: extraction.imageData,
          imageType: extraction.imageType,
          aiDetectedText:
            `${keyFig.significance} ${keyFig.explanation || ""}`.trim(),
        });
      }
    } else {
      console.log(`❌ No semantic match found for ${keyFig.figureNumber}`);

      // Add metadata-only
      matchedFigures.push({
        caption: keyFig.caption,
        pageNumber: keyFig.pageNumber || 1,
        context: keyFig.explanation || keyFig.significance,
        figureNumber: parseInt(keyFig.figureNumber.replace(/[^\d]/g, "")) || 0,
        subpanel: keyFig.figureNumber.match(/[a-z]$/i)?.[0],
        confidence: 0.3,
        aiDetectedText:
          `${keyFig.significance} ${keyFig.explanation || ""}`.trim(),
      });
    }
  }

  return matchedFigures;
}

// ===== MAIN FUNCTION =====

export async function extractPdfWithOpenAI(pdfBuffer: Buffer): Promise<{
  title: string;
  shortExplanation: string;
  summary: string;
  pageCount: number;
  figuresWithImages: OpenAIPdfFigure[];
}> {
  console.log(
    "🤖 Starting OpenAI PDF extraction with EXACT batch script approach..."
  );

  try {
    // Step 1: OpenAI analysis
    console.log("📝 Step 1: Getting key figures and summary from OpenAI...");
    const summaryResult = await generateObject({
      model: openai("gpt-4o"),
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
4. **Key Figures**: The 3-5 most important figures with:
   - EXACT identifiers including sub-panels (e.g., "Figure 1a", "Figure 2b", "Figure 3c", NOT just "Figure 1")
   - Why each figure is significant
   - A detailed explanation of what the figure shows and how to interpret it

CRITICAL: If a figure has multiple sub-panels (like Figure 1a, 1b, 1c), you MUST specify the exact sub-panel (e.g., "Figure 1a") - never just "Figure 1". Look carefully at the paper to identify which specific sub-panel is most important.`,
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

    console.log(
      `✅ OpenAI identified ${summaryResult.object.keyFigures?.length || 0} key figures`
    );

    let figuresWithImages: OpenAIPdfFigure[] = [];

    // Step 2-4: Semantic figure extraction (if figures were identified)
    if (
      summaryResult.object.keyFigures &&
      summaryResult.object.keyFigures.length > 0
    ) {
      try {
        console.log(
          "\n🔧 Step 2: Getting raw Adobe data (EXACTLY like batch script)..."
        );
        const rawAdobeElements = await getRawAdobeElements(pdfBuffer);

        console.log("🎯 Step 3: Creating semantic mappings from raw data...");
        const semanticMappings = createSemanticMappings(rawAdobeElements);
        console.log(`✅ Created ${semanticMappings.size} semantic mappings`);

        console.log(
          "🔍 Step 4: Extracting figures using semantic validation..."
        );
        figuresWithImages = await extractFiguresWithSemanticValidation(
          summaryResult.object.keyFigures,
          rawAdobeElements,
          semanticMappings,
          pdfBuffer
        );

        const imagesFound = figuresWithImages.filter(
          (fig) => fig.imageData
        ).length;
        console.log(
          `🎉 Successfully extracted ${imagesFound}/${summaryResult.object.keyFigures.length} key figures!`
        );
      } catch (figureError) {
        console.error("⚠️ Figure extraction failed:", figureError);

        // Convert to metadata-only figures
        figuresWithImages = summaryResult.object.keyFigures.map((fig) => ({
          caption: fig.caption,
          pageNumber: fig.pageNumber || 1,
          context: fig.explanation || fig.significance,
          figureNumber: parseInt(fig.figureNumber.replace(/[^\d]/g, "")) || 0,
          subpanel: fig.figureNumber.match(/[a-z]$/i)?.[0],
          confidence: 0.6,
          aiDetectedText: `${fig.significance} ${fig.explanation || ""}`.trim(),
        }));
      }
    }

    return {
      title: summaryResult.object.title,
      shortExplanation: summaryResult.object.shortExplanation,
      summary: summaryResult.object.summary,
      pageCount: summaryResult.object.pageCount,
      figuresWithImages,
    };
  } catch (error) {
    console.error("❌ Error in OpenAI PDF extraction:", error);
    throw error;
  }
}
