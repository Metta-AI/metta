#!/usr/bin/env node

/**
 * Extract a specific figure from PDF using Textract coordinates
 *
 * Usage: node extract-figure.js [path-to-pdf]
 */

const pdf2pic = require("pdf2pic");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

// Figure coordinates from Textract (Page 7 - LINE text element)
const FIGURE_DATA = {
  page: 7,
  left: 0.18135808408260345, // 18.1% from left
  top: 0.24006931483745575, // 24.0% from top
  width: 0.2111148089170456, // 21.1% of page width
  height: 0.004524533171206713, // 0.45% of page height (VERY THIN - just text line)
};

async function extractFigure(pdfPath) {
  try {
    console.log(`🔍 Processing PDF: ${pdfPath}`);

    // Check if PDF exists
    if (!fs.existsSync(pdfPath)) {
      throw new Error(`PDF file not found: ${pdfPath}`);
    }

    // Create temp directory for conversion
    const tempDir = "./temp-pdf-extraction";
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir);
    }

    console.log(`📄 Converting page ${FIGURE_DATA.page} to image...`);

    // Convert PDF page to image using pdf2pic
    const convert = pdf2pic.fromPath(pdfPath, {
      density: 300, // High quality (300 DPI)
      saveFilename: "page",
      savePath: tempDir,
      format: "png",
      width: 2000, // Max width for good quality
      height: 2800, // Max height for good quality
    });

    // Convert specific page
    const result = await convert(FIGURE_DATA.page);
    console.log(`✅ Page converted successfully`);

    // Get the generated image path (pdf2pic saves with .1 suffix for single pages)
    const imagePath = path.join(tempDir, `page.${FIGURE_DATA.page}.png`);

    if (!fs.existsSync(imagePath)) {
      throw new Error(`Converted image not found: ${imagePath}`);
    }

    console.log(`📐 Getting image dimensions and cropping figure...`);

    // Load image and get dimensions
    const imageBuffer = fs.readFileSync(imagePath);
    const imageInfo = await sharp(imageBuffer).metadata();

    const pageWidth = imageInfo.width;
    const pageHeight = imageInfo.height;

    console.log(`📊 Page size: ${pageWidth}x${pageHeight} pixels`);

    // Convert relative coordinates to absolute pixels
    const cropLeft = Math.round(FIGURE_DATA.left * pageWidth);
    const cropTop = Math.round(FIGURE_DATA.top * pageHeight);
    const cropWidth = Math.round(FIGURE_DATA.width * pageWidth);
    const cropHeight = Math.round(FIGURE_DATA.height * pageHeight);

    console.log(
      `✂️ Cropping region: ${cropLeft},${cropTop} ${cropWidth}x${cropHeight}`
    );

    // Crop the figure
    const croppedBuffer = await sharp(imageBuffer)
      .extract({
        left: cropLeft,
        top: cropTop,
        width: cropWidth,
        height: cropHeight,
      })
      .png()
      .toBuffer();

    // Save extracted figure
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const outputFile = `extracted-figure-page${FIGURE_DATA.page}-${timestamp}.png`;

    fs.writeFileSync(outputFile, croppedBuffer);
    console.log(`🎯 Figure extracted to: ${outputFile}`);

    // Show figure info
    const figureInfo = await sharp(croppedBuffer).metadata();
    console.log(
      `📸 Figure size: ${figureInfo.width}x${figureInfo.height} pixels`
    );

    // Cleanup temp files
    console.log(`🧹 Cleaning up temporary files...`);
    if (fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
    }
    // Remove temp directory if it exists and is empty
    try {
      if (fs.existsSync(tempDir)) {
        const files = fs.readdirSync(tempDir);
        files.forEach((file) => {
          fs.unlinkSync(path.join(tempDir, file));
        });
        fs.rmdirSync(tempDir);
      }
    } catch (cleanupError) {
      console.warn(
        `⚠️ Could not cleanup temp directory: ${cleanupError.message}`
      );
    }

    console.log(`✅ Figure extraction complete!`);
  } catch (error) {
    console.error("❌ Error extracting figure:", error.message);
    console.error(error.stack);
  }
}

// Main execution
const pdfPath = process.argv[2];

if (!pdfPath) {
  console.log("Usage: node extract-figure.js [path-to-pdf]");
  console.log("");
  console.log("This script extracts a specific figure from page 4 of your PDF");
  console.log("using coordinates detected by AWS Textract.");
  console.log("");
  console.log("Example:");
  console.log("  node extract-figure.js ./my-research-paper.pdf");
  process.exit(1);
}

// Run extraction
extractFigure(pdfPath);
