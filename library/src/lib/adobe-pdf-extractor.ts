import {
  ServicePrincipalCredentials,
  PDFServices,
  MimeType,
  ExtractPDFJob,
  ExtractPDFParams,
  ExtractElementType,
  ExtractPDFResult,
} from "@adobe/pdfservices-node-sdk";
import AdmZip from "adm-zip";
import { readFileSync, writeFileSync, unlinkSync } from "fs";
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
 * Figure with extracted image data
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
 * Adobe PDF element from structured data
 */
interface AdobeElement {
  Path: string;
  Text?: string;
  Page: number;
  Bounds?: [number, number, number, number]; // [x, y, x2, y2]
  ObjectID: number;
}

/**
 * Semantic mapping for figures
 */
interface SemanticMapping {
  objectID: number;
  semanticLabel: string;
  figureNumber: number;
  subpanel?: string;
  confidence: number;
}

/**
 * Figure region detected by Adobe
 */
interface FigureRegion {
  pageNumber: number;
  bounds: [number, number, number, number];
  text?: string; // Any text within the figure region
}

/**
 * Extract PDF content and figures using Adobe PDF Extract API
 */
export async function extractPdfWithAdobe(pdfBuffer: Buffer): Promise<{
  fullText: string;
  figuresWithImages: PdfFigureWithImage[];
  pageCount: number;
}> {
  const clientId = process.env.ADOBE_CLIENT_ID;
  const clientSecret = process.env.ADOBE_CLIENT_SECRET;

  if (!clientId || !clientSecret) {
    throw new Error(
      "Adobe PDF Services credentials not found. Set ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET environment variables."
    );
  }

  console.log("🔄 Starting Adobe PDF extraction...");

  // Create credentials and PDF services client
  const credentials = new ServicePrincipalCredentials({
    clientId: clientId,
    clientSecret: clientSecret,
  });

  const pdfServices = new PDFServices({ credentials });

  // Create temporary file for upload
  const tempFileName = `temp-pdf-${uuidv4()}.pdf`;
  writeFileSync(tempFileName, pdfBuffer);

  try {
    // Upload PDF to Adobe
    console.log("📤 Uploading PDF to Adobe...");
    const inputAsset = await pdfServices.upload({
      readStream: require("fs").createReadStream(tempFileName),
      mimeType: MimeType.PDF,
    });

    // Configure extraction parameters
    const params = new ExtractPDFParams({
      elementsToExtract: [ExtractElementType.TEXT, ExtractElementType.TABLES],
    });

    // Create and submit extraction job
    const job = new ExtractPDFJob({ inputAsset, params });
    console.log("⚙️ Submitting extraction job...");

    const pollingURL = await pdfServices.submit({ job });
    const pdfServicesResponse = await pdfServices.getJobResult({
      pollingURL,
      resultType: ExtractPDFResult,
    });

    console.log("✅ Adobe extraction completed!");

    // Get the result content
    const resultAsset = pdfServicesResponse.result?.resource;
    if (!resultAsset) {
      throw new Error("Adobe PDF Services did not return a result asset");
    }
    const streamAsset = await pdfServices.getContent({ asset: resultAsset });

    // Save result to temporary file
    const tempZipFile = `adobe-results-${uuidv4()}.zip`;
    const outputStream = require("fs").createWriteStream(tempZipFile);
    streamAsset.readStream.pipe(outputStream);

    await new Promise((resolve) => {
      outputStream.on("finish", resolve);
    });

    // Extract and process the ZIP contents
    const zip = new AdmZip(tempZipFile);
    const zipEntries = zip.getEntries();

    // Find the structured data JSON
    const jsonEntry = zipEntries.find((entry: any) =>
      entry.entryName.includes("structuredData.json")
    );

    if (!jsonEntry) {
      throw new Error("No structured data found in Adobe response");
    }

    const structuredData = JSON.parse(jsonEntry.getData().toString("utf8"));
    console.log(
      `📊 Processing ${structuredData.elements?.length || 0} elements...`
    );

    // Extract page count
    const pageCount = structuredData.extended_metadata?.page_count || 0;

    // Process the structured data
    const { fullText, figuresWithImages } = await processAdobeData(
      structuredData,
      pdfBuffer
    );

    // Cleanup
    unlinkSync(tempZipFile);

    console.log(
      `🎉 Extraction complete: ${figuresWithImages.length} figures found`
    );

    return {
      fullText,
      figuresWithImages,
      pageCount,
    };
  } finally {
    // Cleanup temp file
    if (require("fs").existsSync(tempFileName)) {
      unlinkSync(tempFileName);
    }
  }
}

/**
 * Process Adobe structured data to extract text and figures
 */
async function processAdobeData(
  structuredData: any,
  pdfBuffer: Buffer
): Promise<{
  fullText: string;
  figuresWithImages: PdfFigureWithImage[];
}> {
  const elements: AdobeElement[] = structuredData.elements || [];

  // Extract text elements and sort by reading order
  const textElements = elements
    .filter((el) => el.Text && el.Text.trim())
    .sort((a, b) => {
      // Sort by page first
      if (a.Page !== b.Page) return a.Page - b.Page;

      // Then by Y coordinate (top to bottom)
      const aY = a.Bounds ? a.Bounds[1] : 0;
      const bY = b.Bounds ? b.Bounds[1] : 0;
      if (Math.abs(aY - bY) > 5) return bY - aY; // Higher Y first (PDF coordinates)

      // Then by X coordinate (left to right)
      const aX = a.Bounds ? a.Bounds[0] : 0;
      const bX = b.Bounds ? b.Bounds[0] : 0;
      return aX - bX;
    });

  // Reconstruct full text
  const fullText = textElements
    .map((el) => el.Text)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();

  console.log(`📝 Extracted ${fullText.length} characters of text`);

  // Create semantic mappings
  console.log("🎯 Creating semantic mappings...");
  const semanticMappings = createSemanticMappings(elements);
  console.log(`✅ Created ${semanticMappings.size} semantic mappings`);

  // Filter to only actual figures that have semantic mappings
  const validFigures = filterActualFigures(elements).filter((element) =>
    semanticMappings.has(element.ObjectID)
  );

  console.log(
    `📊 Will extract ${validFigures.length} semantically labeled figures`
  );

  // Process figures with semantic labels
  const figuresWithImages = await processSemanticFigures(
    validFigures,
    semanticMappings,
    pdfBuffer
  );

  return {
    fullText,
    figuresWithImages,
  };
}

/**
 * Analyze caption text to extract figure number and subpanel info
 */
function analyzeCaption(captionText: string): {
  figureNumber: number;
  expectedSubpanels: string[];
  isSingleFigure: boolean;
} {
  const figureMatch = captionText.match(/^Figure (\d+):/);
  if (!figureMatch) {
    throw new Error(`Invalid caption format: ${captionText}`);
  }

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

/**
 * Find figure group using structural analysis
 */
function findFigureGroup(
  elements: AdobeElement[],
  captionAnchor: AdobeElement
): AdobeElement[] {
  const captionIndex = elements.findIndex(
    (e) => e.ObjectID === captionAnchor.ObjectID
  );
  if (captionIndex === -1) return [];

  const figureGroup: AdobeElement[] = [];

  // Collect all elements backward from caption until boundary
  for (let i = captionIndex - 1; i >= 0; i--) {
    const element = elements[i];

    // Stop at boundary conditions
    if (
      // Stop at other figure captions
      (element.Text && /^Figure \d+:/.test(element.Text.trim())) ||
      // Stop at paragraph content (boundary detection)
      (element.Path?.includes("/P[") &&
        !element.Path?.includes("/Figure") &&
        element.Text &&
        element.Text.length > 50)
    ) {
      break;
    }

    // Include figure elements and small text elements
    if (
      element.Path?.includes("Figure") ||
      (element.Text && element.Text.length < 20)
    ) {
      figureGroup.unshift(element);
    }
  }

  return figureGroup;
}

/**
 * Check if element is an actual figure (not equation/formula)
 */
function isActualFigure(element: AdobeElement): boolean {
  if (!element.Bounds) return false;
  if (!element.Path?.includes("Figure") || element.Text) return false;

  return true;
}

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
  figureGroup: AdobeElement[],
  actualFigures: AdobeElement[],
  figureNumber: number
): SemanticMapping[] {
  const mappings: SemanticMapping[] = [];

  // Sort figure group by ObjectID to maintain order
  const sortedGroup = [...figureGroup].sort((a, b) => a.ObjectID - b.ObjectID);

  // Create figure-to-label pairs by looking at immediate following elements
  const figureLabelPairs: { figure: AdobeElement; label?: string }[] = [];

  for (let i = 0; i < sortedGroup.length; i++) {
    const element = sortedGroup[i];

    // If this is an actual figure, look for its label
    if (actualFigures.includes(element)) {
      let label: string | undefined;

      // Look at the next few elements to find a subpanel label
      for (let j = i + 1; j < Math.min(i + 3, sortedGroup.length); j++) {
        const nextElement = sortedGroup[j];
        if (nextElement.Text && /^\([a-z]\)/.test(nextElement.Text.trim())) {
          // Extract just the letter from "(a) text" -> "a"
          const match = nextElement.Text.trim().match(/^\(([a-z])\)/);
          if (match) {
            label = match[1];
            break;
          }
        }
        // Stop if we hit another figure or long text
        if (
          actualFigures.includes(nextElement) ||
          (nextElement.Text && nextElement.Text.length > 30)
        ) {
          break;
        }
      }

      figureLabelPairs.push({ figure: element, label });
    }
  }

  // Create mappings based on figure-label pairs
  for (const pair of figureLabelPairs) {
    if (pair.label) {
      mappings.push({
        objectID: pair.figure.ObjectID,
        semanticLabel: `Figure ${figureNumber}${pair.label}`,
        figureNumber: figureNumber,
        subpanel: pair.label,
        confidence: 0.95, // High confidence for explicit figure-label pairing
      });
    } else {
      // Figure without explicit label - assign sequential letter
      const usedLabels = new Set(mappings.map((m) => m.subpanel));
      const subpanelLetters = ["a", "b", "c", "d", "e", "f", "g", "h"];
      const availableLabel = subpanelLetters.find(
        (letter) => !usedLabels.has(letter)
      );

      if (availableLabel) {
        mappings.push({
          objectID: pair.figure.ObjectID,
          semanticLabel: `Figure ${figureNumber}${availableLabel}`,
          figureNumber: figureNumber,
          subpanel: availableLabel,
          confidence: 0.8, // Lower confidence for inferred labeling
        });
      }
    }
  }

  return mappings;
}

/**
 * Create semantic mappings from caption anchors to figure elements
 */
function createSemanticMappings(
  allElements: AdobeElement[]
): Map<number, SemanticMapping> {
  const mappings = new Map<number, SemanticMapping>();

  // Find caption anchors
  const captionAnchors = allElements.filter(
    (element: AdobeElement) =>
      element.Text && /^Figure \d+:/.test(element.Text.trim())
  );

  console.log(
    `📋 Found ${captionAnchors.length} caption anchors for semantic mapping`
  );

  for (const captionAnchor of captionAnchors) {
    try {
      const analysis = analyzeCaption(captionAnchor.Text!);
      const figureGroup = findFigureGroup(allElements, captionAnchor);
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
            confidence: 0.95,
          });
        }
      } else {
        // Multi-panel figure - use structural detection or caption analysis
        if (isStructuralMultiPanel) {
          // Use structural detection to assign subpanel labels
          const subpanelMappings = createStructuralSubpanelMappings(
            figureGroup,
            actualFigures,
            analysis.figureNumber
          );
          for (const mapping of subpanelMappings) {
            mappings.set(mapping.objectID, mapping);
          }
        } else {
          // Use caption-based expected subpanels
          for (
            let i = 0;
            i <
            Math.min(actualFigures.length, analysis.expectedSubpanels.length);
            i++
          ) {
            const figure = actualFigures[i];
            const subpanel = analysis.expectedSubpanels[i];

            mappings.set(figure.ObjectID, {
              objectID: figure.ObjectID,
              semanticLabel: `Figure ${analysis.figureNumber}${subpanel}`,
              figureNumber: analysis.figureNumber,
              subpanel,
              confidence: 0.95,
            });
          }
        }
      }
    } catch (error) {
      console.log(
        `   ❌ Error processing Figure caption: ${(error as any)?.message}`
      );
    }
  }

  return mappings;
}

/**
 * Filter to actual figures using semantic validation
 */
function filterActualFigures(elements: AdobeElement[]): AdobeElement[] {
  // Apply pattern-based filtering like proven approach
  const captionAnchors = elements.filter(
    (element) => element.Text && /^Figure \d+:/.test(element.Text.trim())
  );

  const validatedFigures = new Set<number>();

  for (const captionAnchor of captionAnchors) {
    const figureGroup = findFigureGroup(elements, captionAnchor);
    const actualFigures = figureGroup.filter(isActualFigure);

    for (const figure of actualFigures) {
      validatedFigures.add(figure.ObjectID);
    }
  }

  return elements.filter(
    (element) =>
      element.Bounds &&
      element.Path?.includes("Figure") &&
      !element.Text &&
      validatedFigures.has(element.ObjectID)
  );
}

/**
 * Process semantically mapped figures with image extraction
 */
async function processSemanticFigures(
  validFigures: AdobeElement[],
  semanticMappings: Map<number, SemanticMapping>,
  pdfBuffer: Buffer
): Promise<PdfFigureWithImage[]> {
  const figures: PdfFigureWithImage[] = [];

  // Convert PDF pages to images first
  console.log("🖼️ Converting PDF pages to images...");
  const pageImages = await convertPdfToImages(pdfBuffer);
  console.log(`✅ Converted ${Object.keys(pageImages).length} pages to images`);

  // Process each figure
  const pageImageDimensions: {
    [pageNumber: number]: { width: number; height: number };
  } = {};

  for (const element of validFigures) {
    const mapping = semanticMappings.get(element.ObjectID);
    if (!mapping || !element.Bounds) continue;

    const pageNumber = element.Page;

    // Get page image dimensions if not cached
    if (!pageImageDimensions[pageNumber] && pageImages[pageNumber]) {
      try {
        const tempFile = `/tmp/measure-${uuidv4()}.png`;
        writeFileSync(tempFile, pageImages[pageNumber]);
        const output = execSync(`/usr/bin/gm identify '${tempFile}'`, {
          stdio: "pipe",
        }).toString();
        const match = output.match(/(\d+)x(\d+)/);
        unlinkSync(tempFile);

        if (match) {
          pageImageDimensions[pageNumber] = {
            width: parseInt(match[1]),
            height: parseInt(match[2]),
          };
        }
      } catch (error) {
        console.warn("⚠️ Could not determine image dimensions");
        pageImageDimensions[pageNumber] = { width: 1800, height: 2333 };
      }
    }

    // Extract image data if we have a page image
    let imageData: string | undefined;
    let imageType: string | undefined;

    console.log(`🔍 Processing ${mapping.semanticLabel}:`);
    console.log(`  - Page: ${pageNumber}`);
    console.log(`  - ObjectID: ${element.ObjectID}`);
    console.log(`  - Page image available: ${!!pageImages[pageNumber]}`);

    // Debug specific Figure 1a
    if (mapping.semanticLabel === "Figure 1a") {
      console.log(`🎯 FIGURE 1A DEBUG: This should be the correct extraction!`);
      console.log(`   Expected ObjectID: 1347, Actual: ${element.ObjectID}`);
      console.log(`   Expected Page: 3, Actual: ${pageNumber}`);
      console.log(`   Bounds: [${element.Bounds?.join(", ") || "none"}]`);
    }

    if (pageImages[pageNumber]) {
      try {
        const pageImage = pageImages[pageNumber];
        const pageDims = pageImageDimensions[pageNumber];
        console.log(
          `  - Page image: ${pageImage.length} bytes (${pageDims.width}x${pageDims.height})`
        );

        // Transform PDF coordinates to image coordinates using proven approach
        const pdfWidth = 612; // Standard PDF page width in points
        const pdfHeight = 792; // Standard PDF page height in points

        const scaleX = pageDims.width / pdfWidth;
        const scaleY = pageDims.height / pdfHeight;

        const x = element.Bounds[0] * scaleX;
        const y = (pdfHeight - element.Bounds[3]) * scaleY;
        const width = (element.Bounds[2] - element.Bounds[0]) * scaleX;
        const height = (element.Bounds[3] - element.Bounds[1]) * scaleY;

        const transformedCoords = {
          x: Math.round(x),
          y: Math.round(y),
          width: Math.round(width),
          height: Math.round(height),
        };

        console.log(
          `  - PDF coords: [${element.Bounds.map((b) => b.toFixed(1)).join(", ")}]`
        );
        console.log(
          `  - Scale factors: scaleX=${scaleX.toFixed(3)}, scaleY=${scaleY.toFixed(3)}`
        );
        console.log(
          `  - Page dimensions: ${pageDims.width}x${pageDims.height}`
        );
        console.log(
          `  - Image coords: x=${transformedCoords.x}, y=${transformedCoords.y}, w=${transformedCoords.width}, h=${transformedCoords.height}`
        );

        if (mapping.semanticLabel === "Figure 1a") {
          console.log(`🎯 FIGURE 1A COORDINATE TRANSFORM:`);
          console.log(`   PDF size: ${pdfWidth}x${pdfHeight}`);
          console.log(`   Image size: ${pageDims.width}x${pageDims.height}`);
          console.log(
            `   Raw calculation: x=${element.Bounds[0]} * ${scaleX} = ${element.Bounds[0] * scaleX}`
          );
          console.log(
            `   Raw calculation: y=(${pdfHeight} - ${element.Bounds[3]}) * ${scaleY} = ${(pdfHeight - element.Bounds[3]) * scaleY}`
          );
        }

        const croppedImage = await extractFigureImage(
          pageImage,
          transformedCoords
        );

        console.log(
          `  - Cropped image result: ${!!croppedImage} (${croppedImage ? croppedImage.length : 0} bytes)`
        );

        if (croppedImage) {
          imageData = croppedImage.toString("base64");
          imageType = "png";
          console.log(`  - Base64 length: ${imageData.length} chars`);
        }
      } catch (error) {
        console.warn(
          `⚠️ Failed to extract image for ${mapping.semanticLabel}:`,
          error
        );
      }
    } else {
      console.log(`  ❌ No page image for page ${pageNumber}`);
    }

    // Create figure object with semantic label
    const figure: PdfFigureWithImage = {
      caption: mapping.semanticLabel, // Use semantic label as caption
      pageNumber: pageNumber,
      context: mapping.semanticLabel, // Context is the semantic label
      imageData,
      imageType,
      boundingBox: {
        x: element.Bounds[0],
        y: element.Bounds[1],
        width: element.Bounds[2] - element.Bounds[0],
        height: element.Bounds[3] - element.Bounds[1],
      },
    };

    figures.push(figure);

    const imageStatus = imageData ? "🖼️" : "📌";
    console.log(
      `${imageStatus} Processed ${mapping.semanticLabel} on page ${pageNumber}`
    );
  }

  return figures;
}

/**
 * Convert PDF to page images using direct GraphicsMagick calls
 */
async function convertPdfToImages(
  pdfBuffer: Buffer
): Promise<{ [pageNumber: number]: Buffer }> {
  const pageImages: { [pageNumber: number]: Buffer } = {};
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

    // Convert pages (limit to first 50 pages to avoid too much processing)
    const maxPages = Math.min(50, 41); // We know this PDF has 41 pages

    console.log(`🔄 Converting pages 1-${maxPages}...`);

    for (let pageNum = 1; pageNum <= maxPages; pageNum++) {
      const tempImagePath = `/tmp/pdf-page-${uuidv4()}.png`;

      try {
        console.log(`  📄 Converting page ${pageNum}...`);

        // Use GraphicsMagick to convert PDF page to PNG
        // Use high density + explicit size for optimal quality and correct coordinates
        // CRITICAL: -density must come BEFORE the input file to work properly
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
        pageImages[pageNum - 1] = imageBuffer; // Adobe uses 0-based page numbers
        console.log(
          `  ✅ Page ${pageNum} → index ${pageNum - 1} (${imageBuffer.length} bytes)`
        );

        // Clean up temp image
        unlinkSync(tempImagePath);
      } catch (pageError) {
        console.log(`  ❌ Page ${pageNum} error:`, (pageError as any)?.message);
        // Clean up temp image if it exists
        try {
          unlinkSync(tempImagePath);
        } catch {}
        // Page doesn't exist or other error, break the loop
        break;
      }
    }

    console.log(`📊 Total pages converted: ${Object.keys(pageImages).length}`);
    console.log(
      `📊 Page indices available: [${Object.keys(pageImages).join(", ")}]`
    );

    return pageImages;
  } catch (error) {
    console.error("❌ Error converting PDF to images:", error);
    return {};
  } finally {
    // Clean up temp PDF file
    try {
      unlinkSync(tempPdfPath);
    } catch {}
  }
}

/**
 * Extract figure image from page image using coordinates
 */
async function extractFigureImage(
  pageImageBuffer: Buffer,
  coordinates: { x: number; y: number; width: number; height: number }
): Promise<Buffer | null> {
  try {
    // Use canvas if available, otherwise fall back to GraphicsMagick
    if (canvasAvailable) {
      return await cropWithCanvas(pageImageBuffer, coordinates);
    } else {
      return await cropWithGraphicsMagick(pageImageBuffer, coordinates);
    }
  } catch (error) {
    console.error("❌ Error extracting figure image:", error);
    return null;
  }
}

/**
 * Crop image using Canvas (preferred method)
 */
async function cropWithCanvas(
  imageBuffer: Buffer,
  region: { x: number; y: number; width: number; height: number }
): Promise<Buffer> {
  const image = await loadImage(imageBuffer);

  // Ensure coordinates are within image bounds
  const x = Math.max(0, Math.min(region.x, image.width));
  const y = Math.max(0, Math.min(region.y, image.height));
  const width = Math.min(region.width, image.width - x);
  const height = Math.min(region.height, image.height - y);

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");

  ctx.drawImage(image, x, y, width, height, 0, 0, width, height);

  return canvas.toBuffer("image/png");
}

/**
 * Crop image using GraphicsMagick/ImageMagick (fallback method)
 */
async function cropWithGraphicsMagick(
  imageBuffer: Buffer,
  region: { x: number; y: number; width: number; height: number }
): Promise<Buffer> {
  const tempInputFile = `/tmp/input-${uuidv4()}.png`;
  const tempOutputFile = `/tmp/output-${uuidv4()}.png`;

  try {
    // Write input image to temp file
    writeFileSync(tempInputFile, imageBuffer);

    // Crop using GraphicsMagick
    const cropParams = `${region.width}x${region.height}+${region.x}+${region.y}`;

    try {
      // Try GraphicsMagick first
      execSync(
        `gm convert "${tempInputFile}" -crop ${cropParams} "${tempOutputFile}"`,
        {
          stdio: "pipe",
        }
      );
    } catch (gmError) {
      // Fall back to ImageMagick
      execSync(
        `convert "${tempInputFile}" -crop ${cropParams} "${tempOutputFile}"`,
        {
          stdio: "pipe",
        }
      );
    }

    // Read the cropped image
    const croppedBuffer = readFileSync(tempOutputFile);
    return croppedBuffer;
  } finally {
    // Clean up temp files
    try {
      unlinkSync(tempInputFile);
      unlinkSync(tempOutputFile);
    } catch (cleanupError) {
      // Ignore cleanup errors
    }
  }
}

/**
 * Extract context from figure caption (first sentence or first 100 chars)
 */
function extractContext(caption: string): string {
  // Remove "Figure X:" prefix
  const cleaned = caption.replace(/^Figure\s+\d+[:\.\-\s]*/i, "").trim();

  // Get first sentence or first 100 characters
  const firstSentence = cleaned.split(/[.!?]/)[0];
  if (firstSentence.length <= 100) {
    return firstSentence.trim();
  }

  return cleaned.substring(0, 100).trim() + "...";
}
