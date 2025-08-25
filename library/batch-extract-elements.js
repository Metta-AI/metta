#!/usr/bin/env node

/**
 * Batch extract specific Textract elements by ID
 * 
 * Usage: node batch-extract-elements.js [path-to-pdf] [textract-json-file]
 */

const pdf2pic = require("pdf2pic");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

// Target element IDs to extract
const TARGET_IDS = new Set([
  "8abda5f0-ef2a-4d8b-8b1e-998e2e78443f",
  "616188a5-8c95-4df8-9898-f7936e83b1e1", 
  "b93d6d72-edc5-432d-ad6a-739207ca9c96",
  "629b6039-82e1-4f22-a3fa-00f54ebe7c18",
  "4df5fea6-4b9a-4e47-9323-fe3dbc80a05c",
  "b004f6b3-e23e-4a22-877d-72861ad9754a",
  "0e987e3e-f448-4ab7-b067-6808f3024fbf",
  "943bbf26-2599-433d-a605-b6fa39c09d77",
  "2c4a3bad-57e9-4869-9023-a5f566f8dee4",
  "06f19ecd-26cd-4c7e-8f8d-6469e2d41fda",
  "40f1801e-e6ef-43c1-877d-414865f7a69b",
  "22dee9a6-b2bd-493e-b08c-0dc85c7c833e",
  "af671ef0-0064-455a-aced-b0cfb5044bdf",
  "b6540ab2-2878-4915-9f7a-0bbce5ef9098"
]);

const OUTPUT_DIR = "./textract-extractions";

async function loadTextractData(jsonFile) {
  console.log(`📄 Loading Textract data from: ${jsonFile}`);
  
  const jsonContent = fs.readFileSync(jsonFile, 'utf8');
  const data = JSON.parse(jsonContent);
  
  console.log(`📊 Total blocks in file: ${data.blocks?.length || 0}`);
  
  return data;
}

function findTargetElements(textractData) {
  console.log(`🔍 Filtering for ${TARGET_IDS.size} target element IDs...`);
  
  const blocks = textractData.blocks || [];
  const foundElements = [];
  
  for (const block of blocks) {
    if (TARGET_IDS.has(block.Id)) {
      foundElements.push(block);
      console.log(`✅ Found: ${block.Id} (${block.BlockType}, Page ${block.Page})`);
    }
  }
  
  console.log(`📋 Found ${foundElements.length}/${TARGET_IDS.size} target elements`);
  return foundElements;
}

async function extractElement(pdfPath, element, index) {
  const tempDir = `./temp-extraction-${index}`;
  
  try {
    console.log(`\n🔄 Processing element ${index + 1}: ${element.Id}`);
    console.log(`   Type: ${element.BlockType}, Page: ${element.Page}, Confidence: ${element.Confidence?.toFixed(1)}%`);
    
    // Create temp directory  
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir);
    }
    
    // Convert PDF page to image
    const convert = pdf2pic.fromPath(pdfPath, {
      density: 300,
      saveFilename: "page", 
      savePath: tempDir,
      format: "png",
      width: 2000,
      height: 2800
    });
    
    const result = await convert(element.Page);
    const imagePath = path.join(tempDir, `page.${element.Page}.png`);
    
    if (!fs.existsSync(imagePath)) {
      throw new Error(`Converted image not found: ${imagePath}`);
    }
    
    // Load image and get dimensions
    const imageBuffer = fs.readFileSync(imagePath);
    const imageInfo = await sharp(imageBuffer).metadata();
    
    const pageWidth = imageInfo.width;
    const pageHeight = imageInfo.height;
    
    // Extract bounding box coordinates
    const bbox = element.Geometry.BoundingBox;
    const cropLeft = Math.round(bbox.Left * pageWidth);
    const cropTop = Math.round(bbox.Top * pageHeight);
    const cropWidth = Math.round(bbox.Width * pageWidth);
    const cropHeight = Math.round(bbox.Height * pageHeight);
    
    console.log(`   📐 Crop region: ${cropLeft},${cropTop} ${cropWidth}x${cropHeight}`);
    
    // Crop the element
    const croppedBuffer = await sharp(imageBuffer)
      .extract({
        left: cropLeft,
        top: cropTop,
        width: cropWidth,
        height: cropHeight,
      })
      .png()
      .toBuffer();
    
    // Create output filename
    const shortId = element.Id.substring(0, 8);
    const outputFile = path.join(OUTPUT_DIR, 
      `element-${index + 1}-${element.BlockType}-page${element.Page}-${shortId}.png`
    );
    
    fs.writeFileSync(outputFile, croppedBuffer);
    
    const figureInfo = await sharp(croppedBuffer).metadata();
    console.log(`   📸 Extracted: ${figureInfo.width}x${figureInfo.height}px → ${outputFile}`);
    
    // Add text content if available
    if (element.Text) {
      const textFile = outputFile.replace('.png', '.txt');
      fs.writeFileSync(textFile, `Element ID: ${element.Id}\nType: ${element.BlockType}\nPage: ${element.Page}\nConfidence: ${element.Confidence}\n\nText Content:\n${element.Text}`);
      console.log(`   📝 Text saved: ${textFile}`);
    }
    
    return {
      id: element.Id,
      type: element.BlockType,
      page: element.Page,
      outputFile,
      size: `${figureInfo.width}x${figureInfo.height}`
    };
    
  } catch (error) {
    console.error(`❌ Failed to extract element ${element.Id}:`, error.message);
    return null;
  } finally {
    // Cleanup temp files
    try {
      if (fs.existsSync(tempDir)) {
        const files = fs.readdirSync(tempDir);
        files.forEach((file) => {
          fs.unlinkSync(path.join(tempDir, file));
        });
        fs.rmdirSync(tempDir);
      }
    } catch (cleanupError) {
      console.warn(`⚠️ Could not cleanup temp directory: ${cleanupError.message}`);
    }
  }
}

async function batchExtract(pdfPath, jsonFile) {
  try {
    console.log(`🚀 Starting batch extraction...`);
    console.log(`PDF: ${pdfPath}`);
    console.log(`JSON: ${jsonFile}`);
    console.log(`Output: ${OUTPUT_DIR}\n`);
    
    // Load and filter data
    const textractData = await loadTextractData(jsonFile);
    const targetElements = findTargetElements(textractData);
    
    if (targetElements.length === 0) {
      console.log("❌ No target elements found!");
      return;
    }
    
    // Extract each element
    console.log(`\n🔄 Extracting ${targetElements.length} elements...`);
    const results = [];
    
    for (let i = 0; i < targetElements.length; i++) {
      const result = await extractElement(pdfPath, targetElements[i], i);
      if (result) {
        results.push(result);
      }
    }
    
    // Summary
    console.log(`\n✅ Batch extraction complete!`);
    console.log(`📊 Successfully extracted ${results.length}/${targetElements.length} elements`);
    
    if (results.length > 0) {
      console.log(`\n📋 Extraction Summary:`);
      results.forEach((result, i) => {
        console.log(`   ${i + 1}. ${result.type} (Page ${result.page}) → ${result.size}`);
      });
    }
    
  } catch (error) {
    console.error('❌ Batch extraction failed:', error.message);
    console.error(error.stack);
  }
}

// Main execution
const pdfPath = process.argv[2];
const jsonFile = process.argv[3];

if (!pdfPath || !jsonFile) {
  console.log('Usage: node batch-extract-elements.js [path-to-pdf] [textract-json-file]');
  console.log('');
  console.log('This script extracts specific Textract elements by their IDs.');
  console.log('');
  console.log('Example:');
  console.log('  node batch-extract-elements.js ./paper.pdf ./textract-raw-blocks-*.json');
  process.exit(1);
}

// Run batch extraction
batchExtract(pdfPath, jsonFile);


