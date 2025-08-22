// ✅ NEW: Hybrid OpenAI + Adobe PDF extraction following batch script approach EXACTLY
// 📝 Step 1: OpenAI identifies key figures + provides summary (generateObject + zod)
// 🔧 Step 2: Get raw Adobe elements (same as debug-adobe-raw.ts)
// 🎯 Step 3: Create semantic mappings (same as batch script) OR use LLM-based object selection
// 🔍 Step 4: Extract ONLY semantically validated figures (same as batch script)
// 💾 Step 5: Save extracted figures with correct coordinates
// 🏆 Result: Same proven approach as working batch script
//
// 🆕 LLM-BASED ADOBE OBJECT SELECTION:
// Set environment variable USE_LLM_ADOBE_SELECTION=true to use LLM for selecting
// Adobe objects instead of the fragile semantic mapping logic. The LLM receives
// the desired figure identifiers from OpenAI vision analysis and raw Adobe data,
// then intelligently selects the appropriate objects for each figure.

import { generateObject } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
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

/**
 * Schema for LLM-based Adobe object selection
 */
const AdobeObjectSelectionSchema = z.object({
  selections: z.array(
    z.object({
      figureIdentifier: z
        .string()
        .describe(
          "The exact figure identifier (e.g., 'Figure 1', 'Figure 2a', 'Figure 2b')"
        ),
      selectedObjectID: z
        .number()
        .describe(
          "The ObjectID of the Adobe element that corresponds to this figure"
        ),
      confidence: z
        .enum(["high", "medium", "low"])
        .describe("Confidence level in this selection"),
      reasoning: z
        .string()
        .describe(
          "Brief explanation of why this Adobe element was selected for this figure"
        ),
    })
  ),
  globalReasoning: z
    .string()
    .describe(
      "Overall explanation of the selection process and any challenges encountered"
    ),
});

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
          .describe(
            "Page number where this figure appears (required for efficient processing)"
          ),
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

// ===== RAW PDF DATA EXTRACTION (ADOBE REPLACED WITH AWS TEXTRACT) =====

/**
 * AWS Textract PDF element (mapped from Textract Block)
 */
interface TextractElement {
  ObjectID: number;
  Page: number;
  Path?: string;
  Text?: string;
  Bounds?: number[];
  BlockType: string;
  Confidence?: number;
}

/**
 * AWS Textract implementation - replaces unreliable Adobe PDF Services
 */
async function getRawTextractElements(
  pdfBuffer: Buffer
): Promise<AdobeElement[]> {
  console.log(
    "🔍 Using AWS Textract for PDF element extraction (replacing Adobe)..."
  );

  // Detailed PDF validation (same as before)
  console.log("🔬 Analyzing PDF structure for Textract processing...");

  const pdfSize = pdfBuffer.length;
  console.log(
    `📏 PDF size: ${pdfSize} bytes (${Math.round(pdfSize / 1024)} KB)`
  );

  // Check PDF header
  const pdfHeader = pdfBuffer.slice(0, 8).toString();
  const pdfVersion = pdfHeader.match(/%PDF-(\d\.\d)/)?.[1];
  console.log(`📄 PDF version: ${pdfVersion || "unknown"}`);

  if (!pdfHeader.startsWith("%PDF")) {
    throw new Error(`Invalid PDF header: ${pdfHeader}`);
  }

  // Basic content analysis
  const pdfString = pdfBuffer.toString("binary");
  const hasImages =
    pdfString.includes("/Image") || pdfString.includes("/DCTDecode");
  const hasJavaScript =
    pdfString.includes("/JavaScript") || pdfString.includes("/JS");

  console.log(`📊 PDF analysis for Textract:`);
  console.log(`   - Has images: ${hasImages}`);
  console.log(
    `   - Has JavaScript: ${hasJavaScript} ${hasJavaScript ? "(Textract handles this fine)" : ""}`
  );
  console.log(
    `   - Size: ${Math.round(pdfSize / (1024 * 1024))}MB ${pdfSize > 5 * 1024 * 1024 ? "(will use S3)" : "(direct upload)"}`
  );

  try {
    const textractElements = await processWithTextract(pdfBuffer, pdfSize);

    // Convert Textract elements to Adobe-compatible format
    const adobeElements: AdobeElement[] = textractElements.map((elem) => ({
      ObjectID: elem.ObjectID,
      Page: elem.Page,
      Path: elem.Path,
      Text: elem.Text,
      Bounds: elem.Bounds,
    }));

    console.log(
      `✅ Textract extracted ${adobeElements.length} elements successfully`
    );
    return adobeElements;
  } catch (error: any) {
    console.error("❌ AWS Textract processing failed:", error);
    throw new Error(`AWS Textract failed: ${error.message}`);
  }
}

async function processWithTextract(
  pdfBuffer: Buffer,
  pdfSize: number
): Promise<TextractElement[]> {
  const { TextractClient, AnalyzeDocumentCommand } = await import(
    "@aws-sdk/client-textract"
  );
  const { S3Client, PutObjectCommand, DeleteObjectCommand } = await import(
    "@aws-sdk/client-s3"
  );
  const { v4: uuidv4 } = await import("uuid");

  // Initialize AWS clients
  const textractClient = new TextractClient({
    region: process.env.AWS_REGION || "us-east-1",
  });

  let s3Client: any = null;
  let bucketName = "";
  let s3Key = "";

  try {
    // For files > 5MB, use S3
    if (pdfSize > 5 * 1024 * 1024) {
      console.log("📦 Large PDF - using S3 + Textract...");

      s3Client = new S3Client({
        region: process.env.AWS_REGION || "us-east-1",
      });

      bucketName = process.env.S3_BUCKET || "metta-pdf-processing";
      s3Key = `temp-pdfs/${uuidv4()}.pdf`;

      console.log(
        `⬆️ Uploading ${Math.round(pdfSize / (1024 * 1024))}MB PDF to S3: s3://${bucketName}/${s3Key}`
      );

      await s3Client.send(
        new PutObjectCommand({
          Bucket: bucketName,
          Key: s3Key,
          Body: pdfBuffer,
          ContentType: "application/pdf",
        })
      );

      console.log("✅ S3 upload completed");
    }

    // Prepare Textract request
    const textractRequest: any = {
      FeatureTypes: ["LAYOUT", "TABLES"], // Extract layout and table information
    };

    if (pdfSize > 5 * 1024 * 1024) {
      // Use S3 document location
      textractRequest.Document = {
        S3Object: {
          Bucket: bucketName,
          Name: s3Key,
        },
      };
    } else {
      // Use direct bytes
      textractRequest.Document = {
        Bytes: pdfBuffer,
      };
    }

    console.log("🤖 Running AWS Textract analysis...");
    const startTime = Date.now();

    const response = await textractClient.send(
      new AnalyzeDocumentCommand(textractRequest)
    );

    const processingTime = Date.now() - startTime;
    console.log(`✅ Textract completed in ${processingTime}ms`);

    // Process Textract blocks
    const blocks = response.Blocks || [];
    console.log(`📊 Textract found ${blocks.length} blocks`);

    const elements: TextractElement[] = [];
    let objectIdCounter = 1000; // Start high to avoid conflicts

    for (const block of blocks) {
      // Convert Textract block to our element format
      const element: TextractElement = {
        ObjectID: objectIdCounter++,
        Page: block.Page || 1,
        BlockType: block.BlockType || "UNKNOWN",
        Confidence: block.Confidence,
      };

      // Add text if available
      if (block.Text) {
        element.Text = block.Text;
      }

      // Add bounding box if available (convert from Textract format)
      if (block.Geometry?.BoundingBox) {
        const bbox = block.Geometry.BoundingBox;
        // Textract uses normalized coordinates (0-1), convert to points (assuming ~600pt page width)
        const pageWidth = 612; // Standard PDF page width in points
        const pageHeight = 792; // Standard PDF page height in points

        if (
          bbox.Left !== undefined &&
          bbox.Top !== undefined &&
          bbox.Width !== undefined &&
          bbox.Height !== undefined
        ) {
          element.Bounds = [
            bbox.Left * pageWidth,
            bbox.Top * pageHeight,
            (bbox.Left + bbox.Width) * pageWidth,
            (bbox.Top + bbox.Height) * pageHeight,
          ];
        }
      }

      // Create Path for figure-like elements
      if (
        block.BlockType === "LAYOUT_FIGURE" ||
        (block.BlockType === "LINE" &&
          block.Text?.toLowerCase().includes("figure"))
      ) {
        element.Path = `Figure_${element.ObjectID}`;
      }

      elements.push(element);
    }

    // Cleanup S3 file if used
    if (s3Client && bucketName && s3Key) {
      console.log("🧹 Cleaning up S3 temporary file...");
      try {
        await s3Client.send(
          new DeleteObjectCommand({
            Bucket: bucketName,
            Key: s3Key,
          })
        );
      } catch (cleanupError) {
        console.warn("⚠️ Failed to cleanup S3 file:", cleanupError);
      }
    }

    return elements;
  } catch (error) {
    // Cleanup on error
    if (s3Client && bucketName && s3Key) {
      try {
        await s3Client.send(
          new DeleteObjectCommand({
            Bucket: bucketName,
            Key: s3Key,
          })
        );
      } catch (cleanupError) {
        console.warn("⚠️ Failed to cleanup S3 file after error:", cleanupError);
      }
    }
    throw error;
  }
}

/**
 * Retry operation with exponential backoff (kept for any remaining edge cases)
 */
async function retryAdobeOperation<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 2000
): Promise<T> {
  let lastError: any;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`🔄 Adobe operation attempt ${attempt}/${maxRetries}`);
      return await operation();
    } catch (error: any) {
      lastError = error;

      // Don't retry on certain errors
      const errorMessage = error.message || "";
      if (
        errorMessage.includes("encrypted") ||
        errorMessage.includes("password")
      ) {
        console.log("❌ PDF encryption error - not retrying");
        throw error;
      }

      if (attempt === maxRetries) {
        console.log(`❌ Adobe operation failed after ${maxRetries} attempts`);
        throw error;
      }

      const delay = baseDelay * Math.pow(2, attempt - 1); // Exponential backoff
      console.warn(
        `⚠️ Adobe operation failed (attempt ${attempt}): ${errorMessage}`
      );
      console.log(`⏳ Retrying in ${delay}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}

async function getRawAdobeElements(pdfBuffer: Buffer): Promise<AdobeElement[]> {
  // ⚠️ DEPRECATED: Adobe is unreliable - now using Textract
  console.warn("⚠️ Adobe function called - redirecting to AWS Textract");
  return await getRawTextractElements(pdfBuffer);
}

// ===== FIGURE EXTRACTION (SAME AS BATCH SCRIPT) =====

// TODO: Fix duplicate function - temporarily commented out the broken version
/*
async function extractFigureFromElement_BROKEN(
  element: AdobeElement,
    console.error(
      "❌ PDF appears to be encrypted/password-protected - Adobe will fail"
    );
    throw new Error(
      "PDF is encrypted or password-protected. Adobe PDF Services cannot process encrypted PDFs."
    );
  }

  // Check PDF version
  const pdfHeader = pdfBuffer.slice(0, 8).toString();
  const pdfVersion = pdfHeader.match(/%PDF-(\d\.\d)/)?.[1];
  console.log(`📄 PDF version: ${pdfVersion || "unknown"}`);

  // Size limits (Adobe has processing limits)
  const maxSize = 100 * 1024 * 1024; // 100MB should be safe
  if (pdfSize > maxSize) {
    console.warn(
      `⚠️ PDF is large (${Math.round(pdfSize / (1024 * 1024))}MB) - may cause Adobe processing issues`
    );
  }

  // Check for suspicious content that might cause Adobe issues
  const hasImages =
    pdfString.includes("/Image") ||
    pdfString.includes("/DCTDecode") ||
    pdfString.includes("/JPXDecode");
  const hasComplexGraphics =
    pdfString.includes("/DeviceN") || pdfString.includes("/Separation");
  const hasJavaScript =
    pdfString.includes("/JavaScript") || pdfString.includes("/JS");

  console.log(`📊 PDF analysis:`);
  console.log(`   - Has images: ${hasImages}`);
  console.log(`   - Complex graphics: ${hasComplexGraphics}`);
  console.log(`   - JavaScript: ${hasJavaScript}`);
  console.log(`   - Encrypted: ${isEncrypted}`);

  // Warn about potential issues
  if (hasJavaScript) {
    console.warn(
      "⚠️ PDF contains JavaScript - this often causes Adobe processing failures"
    );
    console.warn(
      "   Adobe PDF Services may reject PDFs with interactive content"
    );
  }

  if (pdfSize > 10 * 1024 * 1024) {
    console.warn(
      `⚠️ Large PDF (${Math.round(pdfSize / (1024 * 1024))}MB) - Adobe processing may be slow or fail`
    );
  }

  // Use retry mechanism for Adobe processing
  return await retryAdobeOperation(async () => {
    return await processWithAdobe(pdfBuffer, pdfSize);
  });
}

async function processWithAdobe(
  pdfBuffer: Buffer,
  pdfSize: number
): Promise<AdobeElement[]> {
  console.log("💾 Writing PDF to temporary file for Adobe...");

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

    // Try to configure client with longer timeouts and better settings
    const clientConfig = {
      credentials,
      // Attempt to increase timeout settings if supported
      connectTimeout: 120000, // 2 minutes
      readTimeout: 300000, // 5 minutes
    };

    const pdfServices = new PDFServices(clientConfig);

    // Write PDF with validation
    console.log(
      `📝 Writing ${pdfSize} bytes to temporary file: ${tempFileName}`
    );
    writeFileSync(tempFileName, pdfBuffer);

    // Verify file was written correctly
    if (!existsSync(tempFileName)) {
      throw new Error(`Failed to create temporary file: ${tempFileName}`);
    }

    const tempFileSize = statSync(tempFileName).size;
    if (tempFileSize !== pdfSize) {
      throw new Error(
        `Temporary file size mismatch: expected ${pdfSize}, got ${tempFileSize}`
      );
    }

    console.log(
      `✅ Temporary file created successfully: ${tempFileSize} bytes`
    );

    const inputAsset = await pdfServices.upload({
      readStream: createReadStream(tempFileName),
      mimeType: MimeType.PDF,
    });

    const params = new ExtractPDFParams({
      elementsToExtract: [ExtractElementType.TEXT, ExtractElementType.TABLES],
    });

    const job = new ExtractPDFJob({ inputAsset, params });

    // Add longer timeout for Adobe PDF Services (large PDFs with JavaScript need more time)
    console.log(
      "⏳ Submitting to Adobe PDF Services (this may take 2-5 minutes for large/complex PDFs)..."
    );
    const submitPromise = pdfServices.submit({ job });
    const pollingURL = (await Promise.race([
      submitPromise,
      new Promise((_, reject) =>
        setTimeout(
          () => reject(new Error("Adobe submit timeout after 180 seconds")),
          180000 // 3 minutes for submit
        )
      ),
    ])) as string;

    console.log("🔄 Polling Adobe for results...");
    const resultPromise = pdfServices.getJobResult({
      pollingURL,
      resultType: ExtractPDFResult,
    });
    const pdfServicesResponse = (await Promise.race([
      resultPromise,
      new Promise((_, reject) =>
        setTimeout(
          () => reject(new Error("Adobe polling timeout after 300 seconds")),
          300000 // 5 minutes for polling
        )
      ),
    ])) as any;

    console.log("💾 Downloading extraction results...");
    const resultAsset = (pdfServicesResponse as any).result?.resource;
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
  } catch (adobeError: any) {
    console.error("❌ Adobe PDF Services error:", adobeError);

    // Enhanced error analysis
    const errorCode =
      adobeError._errorCode || adobeError.errorCode || "UNKNOWN";
    const statusCode =
      adobeError._statusCode || adobeError.statusCode || "UNKNOWN";
    const trackingId =
      adobeError._requestTrackingId || adobeError.requestTrackingId || "NONE";

    console.error(`📋 Adobe Error Details:`);
    console.error(`   - Error Code: ${errorCode}`);
    console.error(`   - Status Code: ${statusCode}`);
    console.error(`   - Tracking ID: ${trackingId}`);
    console.error(`   - Message: ${adobeError.message || "No message"}`);

    // Common Adobe error analysis
    if (errorCode === "ERROR" && statusCode === 500) {
      console.error("🔍 Analysis: Adobe internal error - possible causes:");
      console.error("   - PDF structure not compatible with Adobe extraction");
      console.error("   - PDF contains complex elements Adobe cannot parse");
      console.error("   - PDF may be corrupted or have unusual encoding");
      console.error("   - Adobe service experiencing issues");

      // Re-analyze the PDF for specific issues
      const pdfString = pdfBuffer.toString("binary");
      const hasComplexForms =
        pdfString.includes("/AcroForm") || pdfString.includes("/XFA");
      const hasAnnotations = pdfString.includes("/Annot");
      const hasEmbeddedFiles = pdfString.includes("/EmbeddedFile");
      const hasTransparency =
        pdfString.includes("/SMask") || pdfString.includes("/CA");

      console.error("🔍 Extended PDF analysis:");
      console.error(`   - Has forms: ${hasComplexForms}`);
      console.error(`   - Has annotations: ${hasAnnotations}`);
      console.error(`   - Has embedded files: ${hasEmbeddedFiles}`);
      console.error(`   - Has transparency: ${hasTransparency}`);

      if (hasComplexForms) {
        console.error("⚠️ PDF contains forms - Adobe may struggle with these");
      }
      if (hasEmbeddedFiles) {
        console.error(
          "⚠️ PDF contains embedded files - potential processing issue"
        );
      }
    }

    // Save problematic PDF for debugging (only in development)
    if (process.env.NODE_ENV === "development") {
      const debugFileName = `debug-adobe-error-${Date.now()}.pdf`;
      writeFileSync(debugFileName, pdfBuffer);
      console.error(`💾 Saved problematic PDF for debugging: ${debugFileName}`);
    }

    throw new Error(
      `Adobe PDF Services failed: ${errorCode} - ${adobeError.message || "Internal error"}. Tracking ID: ${trackingId}`
    );
  } finally {
    if (existsSync(tempFileName)) {
      unlinkSync(tempFileName);
    }
  }
}
*/

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

// ===== LLM-BASED ADOBE OBJECT SELECTION =====

/**
 * Use LLM to select appropriate Adobe objects for desired figures
 * Alternative to the fragile createSemanticMappings logic
 */
async function selectAdobeObjectsWithLLM(
  keyFigures: any[],
  rawAdobeElements: AdobeElement[]
): Promise<Map<number, SemanticMapping>> {
  console.log(
    `🤖 Using LLM to select Adobe objects for ${keyFigures.length} desired figures...`
  );

  // Extract unique page numbers from keyFigures to limit scope
  const figurePages = new Set<number>();
  keyFigures.forEach((fig) => {
    if (fig.pageNumber && typeof fig.pageNumber === "number") {
      figurePages.add(fig.pageNumber);
      // Also include adjacent pages for context (figures might span pages)
      figurePages.add(Math.max(1, fig.pageNumber - 1));
      figurePages.add(fig.pageNumber + 1);
    }
  });

  console.log(
    `📄 Filtering Adobe elements to pages: ${Array.from(figurePages).sort().join(", ")}`
  );

  // Filter rawAdobeElements to only include elements from relevant pages
  const filteredAdobeElements =
    figurePages.size > 0
      ? rawAdobeElements.filter((el) => figurePages.has(el.Page))
      : rawAdobeElements; // Fallback if no page numbers available

  console.log(
    `🔍 Reduced Adobe elements from ${rawAdobeElements.length} to ${filteredAdobeElements.length} (${Math.round((filteredAdobeElements.length / rawAdobeElements.length) * 100)}%)`
  );

  // Prepare a concise representation of Adobe elements for the LLM
  const elementsForLLM = filteredAdobeElements.map((el) => ({
    objectID: el.ObjectID,
    page: el.Page,
    path: el.Path,
    text: el.Text?.substring(0, 100), // Truncate long text
    bounds: el.Bounds,
    hasText: !!el.Text,
    isFigureElement: el.Path?.includes("Figure") || false,
    elementType: el.Text
      ? "text"
      : el.Path?.includes("Figure")
        ? "figure"
        : "other",
  }));

  // Create the prompt for the LLM
  const desiredFigures = keyFigures.map((fig) => ({
    figureNumber: fig.figureNumber,
    pageNumber: fig.pageNumber,
    caption: fig.caption,
  }));
  const prompt = `You are analyzing PDF elements extracted by Adobe to select the correct objects that correspond to specific figure identifiers.

DESIRED FIGURES: ${JSON.stringify(desiredFigures, null, 2)}

ADOBE ELEMENTS AVAILABLE:
${JSON.stringify(elementsForLLM, null, 2)}

TASK:
For each desired figure identifier (like "Figure 1", "Figure 2a", "Figure 2b"), you need to select the Adobe element ObjectID that best corresponds to that figure.

GUIDELINES:
1. Look for elements with Path containing "Figure" - these are likely the actual figure objects
2. Elements with Text are usually captions/labels, not the figure images themselves
3. Match page numbers: Each desired figure specifies its page number - select elements from that page
4. Figure elements should have bounds (coordinates) and be reasonably sized
5. For multi-panel figures (like "Figure 2a", "Figure 2b"), look for separate objects that are spatially arranged
6. Single figures (like "Figure 1") should correspond to one main figure object
7. Larger bounding boxes often indicate main figure content vs small decorative elements
8. The Adobe elements have already been filtered to relevant pages, so focus on finding the right objects within those pages

For each desired figure, select the ObjectID that most likely represents that specific figure content.`;

  try {
    const result = await generateObject({
      model: anthropic("claude-3-5-sonnet-20241022"),
      temperature: 0.1,
      messages: [
        {
          role: "system",
          content:
            "You are an expert at analyzing PDF structure and selecting the correct figure objects from Adobe PDF extraction data. You must return a valid JSON object matching the required schema.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      schema: AdobeObjectSelectionSchema,
    });

    console.log(
      `✅ LLM selected ${result.object.selections.length} Adobe objects`
    );
    console.log(`🧠 LLM reasoning: ${result.object.globalReasoning}`);

    // Convert LLM results to SemanticMapping format
    const mappings = new Map<number, SemanticMapping>();

    for (const selection of result.object.selections) {
      const figureNumber =
        parseInt(selection.figureIdentifier.replace(/[^\d]/g, "")) || 0;
      const subpanel = selection.figureIdentifier
        .match(/[a-z]$/i)?.[0]
        ?.toLowerCase();

      mappings.set(selection.selectedObjectID, {
        objectID: selection.selectedObjectID,
        semanticLabel: selection.figureIdentifier,
        figureNumber,
        subpanel,
        confidence:
          selection.confidence === "high"
            ? 0.95
            : selection.confidence === "medium"
              ? 0.85
              : 0.75,
      });

      console.log(
        `  📍 ${selection.figureIdentifier} → ObjectID ${selection.selectedObjectID} (${selection.confidence}) - ${selection.reasoning}`
      );
    }

    return mappings;
  } catch (error) {
    console.error("❌ LLM Adobe object selection failed:", error);
    throw error;
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
        const filename = `${keyFig.figureNumber.replace(/[^a-zA-Z0-9]/g, "_")}.png`;
        const imageSize = Math.round((extraction.imageData.length * 3) / 4); // Base64 to bytes conversion
        console.log(`✅ Extracted: ${filename} (${imageSize} bytes, virtual)`);

        // No file system write - keeping everything in memory as base64

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
    // Validate PDF buffer
    console.log(`📄 Validating PDF buffer (${pdfBuffer.length} bytes)...`);
    if (pdfBuffer.length === 0) {
      throw new Error("PDF buffer is empty");
    }

    // Check PDF header
    const pdfHeader = pdfBuffer.slice(0, 8).toString();
    if (!pdfHeader.startsWith("%PDF")) {
      console.error(`❌ Invalid PDF header: "${pdfHeader}"`);
      console.error(
        `❌ First 50 bytes: ${pdfBuffer.slice(0, 50).toString("hex")}`
      );
      throw new Error(`Invalid PDF: header is "${pdfHeader}", expected "%PDF"`);
    }

    console.log(`✅ Valid PDF detected (version: ${pdfHeader})`);

    // Step 1: Anthropic analysis
    console.log("📝 Step 1: Getting key figures and summary from Anthropic...");
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
4. **Key Figures**: The 3-5 most important figures with:
   - EXACT identifiers including sub-panels (e.g., "Figure 1a", "Figure 2b", "Figure 3c", NOT just "Figure 1")
   - PAGE NUMBER where the figure appears (this is critical for processing)
   - Why each figure is significant
   - A detailed explanation of what the figure shows and how to interpret it

CRITICAL:
- If a figure has multiple sub-panels (like Figure 1a, 1b, 1c), you MUST specify the exact sub-panel (e.g., "Figure 1a") - never just "Figure 1"
- You MUST include the page number for each figure
- Look carefully at the paper to identify which specific sub-panel is most important and on which page it appears`,
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
      `✅ Anthropic identified ${summaryResult.object.keyFigures?.length || 0} key figures`
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

        // Configuration flag to use LLM-based object selection
        const useLlmSelection = process.env.USE_LLM_ADOBE_SELECTION === "true";

        let semanticMappings: Map<number, SemanticMapping>;

        if (useLlmSelection) {
          console.log("🤖 Using LLM-based Adobe object selection...");
          try {
            semanticMappings = await selectAdobeObjectsWithLLM(
              summaryResult.object.keyFigures,
              rawAdobeElements
            );
            console.log(
              `✅ LLM created ${semanticMappings.size} object selections`
            );
          } catch (error) {
            console.warn(
              "⚠️ LLM selection failed, falling back to traditional semantic mapping:",
              error
            );
            semanticMappings = createSemanticMappings(rawAdobeElements);
            console.log(
              `✅ Fallback created ${semanticMappings.size} semantic mappings`
            );
          }
        } else {
          console.log("🎯 Using traditional semantic mapping...");
          semanticMappings = createSemanticMappings(rawAdobeElements);
          console.log(`✅ Created ${semanticMappings.size} semantic mappings`);
        }

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
    console.error("❌ Error in Anthropic PDF extraction:", error);
    throw error;
  }
}
