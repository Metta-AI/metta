#!/usr/bin/env tsx

/**
 * arXiv Paper Fetcher Script
 *
 * This script fetches paper metadata from arXiv given a URL or arXiv ID.
 * It can be used both as a command-line tool and imported as a module.
 *
 * Usage as CLI:
 *   tsx scripts/fetch-arxiv-paper.ts https://arxiv.org/abs/2204.11674
 *   tsx scripts/fetch-arxiv-paper.ts 2204.11674
 *
 * Usage as module:
 *   import { fetchArxivPaper } from './scripts/fetch-arxiv-paper';
 *   const paperData = await fetchArxivPaper('2204.11674');
 */

import * as dotenv from "dotenv";

// Load environment variables
dotenv.config();
dotenv.config({ path: ".env.local", override: true });

// XML parsing library for arXiv RSS feed
import { XMLParser } from "fast-xml-parser";

/**
 * Interface defining the structure of arXiv paper data
 */
export interface ArxivPaperData {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  categories: string[];
  publishedDate: string;
  updatedDate: string;
  doi?: string;
  journalRef?: string;
  primaryCategory: string;
  arxivUrl: string;
  pdfUrl: string;
  summary: string;
}

/**
 * Interface for the raw arXiv API response
 */
interface ArxivApiResponse {
  feed: {
    entry: {
      id: string;
      title: string;
      summary: string;
      published: string;
      updated: string;
      author: { name: string } | Array<{ name: string }>;
      category:
        | { _: string; term: string }
        | Array<{ _: string; term: string }>;
      link:
        | { href: string; title?: string; type?: string }
        | Array<{ href: string; title?: string; type?: string }>;
      "arxiv:doi"?: { _: string };
      "arxiv:journal_ref"?: { _: string };
      "arxiv:primary_category"?: { term: string };
    };
  };
}

/**
 * Extracts arXiv ID from various input formats
 *
 * @param input - Can be a full URL, arXiv ID, or arXiv ID with version
 * @returns Clean arXiv ID without version suffix
 */
export function extractArxivId(input: string): string {
  // Remove any whitespace
  const cleanInput = input.trim();

  // Handle full URLs
  if (cleanInput.includes("arxiv.org")) {
    const urlMatch = cleanInput.match(/arxiv\.org\/abs\/(\d+\.\d+)/);
    if (urlMatch) {
      return urlMatch[1];
    }
  }

  // Handle arXiv IDs (with or without version suffix)
  const idMatch = cleanInput.match(/^(\d+\.\d+)/);
  if (idMatch) {
    return idMatch[1];
  }

  throw new Error(
    `Invalid arXiv identifier: ${input}. Expected format: 2204.11674 or https://arxiv.org/abs/2204.11674`
  );
}

/**
 * Fetches paper metadata from arXiv API
 *
 * @param arxivId - The arXiv ID (e.g., "2204.11674")
 * @returns Promise resolving to structured paper data
 */
export async function fetchArxivPaper(
  arxivId: string
): Promise<ArxivPaperData> {
  try {
    // Clean the arXiv ID
    const cleanId = extractArxivId(arxivId);

    // Construct the arXiv API URL
    const apiUrl = `http://export.arxiv.org/api/query?id_list=${cleanId}`;

    console.log(`🔍 Fetching data for arXiv:${cleanId}...`);

    // Fetch data from arXiv API
    const response = await fetch(apiUrl);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const xmlData = await response.text();

    // Parse XML response
    const parser = new XMLParser({
      ignoreAttributes: false,
      attributeNamePrefix: "",
      textNodeName: "_",
      parseAttributeValue: false,
    });

    const parsedData = parser.parse(xmlData) as ArxivApiResponse;

    // Check if we got a valid response
    if (!parsedData.feed?.entry) {
      throw new Error(`No paper found for arXiv:${cleanId}`);
    }

    const entry = parsedData.feed.entry;

    // Extract authors
    const authors = Array.isArray(entry.author)
      ? entry.author.map((author) => author.name)
      : [entry.author.name];

    // Extract categories
    const categories = Array.isArray(entry.category)
      ? entry.category.map((cat) => cat.term)
      : [entry.category.term];

    // Extract links
    const links = Array.isArray(entry.link) ? entry.link : [entry.link];
    let arxivUrl = links.find((link) => link.title === "abs")?.href || "";
    const pdfUrl =
      links.find((link) => link.type === "application/pdf")?.href || "";

    // If no abstract URL found, construct it from the arXiv ID
    if (!arxivUrl && cleanId) {
      arxivUrl = `https://arxiv.org/abs/${cleanId}`;
    }

    // Extract DOI and journal reference if available
    const doi = parsedData.feed.entry["arxiv:doi"]?._;
    const journalRef = parsedData.feed.entry["arxiv:journal_ref"]?._;
    const primaryCategory =
      parsedData.feed.entry["arxiv:primary_category"]?.term || categories[0];

    // Construct the structured response
    const paperData: ArxivPaperData = {
      id: cleanId,
      title: entry.title.replace(/\s+/g, " ").trim(), // Clean up whitespace
      abstract: entry.summary.replace(/\s+/g, " ").trim(), // Clean up whitespace
      authors,
      categories,
      publishedDate: entry.published,
      updatedDate: entry.updated,
      doi,
      journalRef,
      primaryCategory,
      arxivUrl,
      pdfUrl,
      summary: entry.summary.replace(/\s+/g, " ").trim(), // Alias for abstract
    };

    console.log(`✅ Successfully fetched data for "${paperData.title}"`);

    return paperData;
  } catch (error) {
    console.error(
      `❌ Error fetching arXiv paper: ${error instanceof Error ? error.message : "Unknown error"}`
    );
    throw error;
  }
}

/**
 * Main function for command-line usage
 */
async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log("📚 arXiv Paper Fetcher");
    console.log("");
    console.log("Usage:");
    console.log("  tsx scripts/fetch-arxiv-paper.ts <arxiv-url-or-id>");
    console.log("");
    console.log("Examples:");
    console.log(
      "  tsx scripts/fetch-arxiv-paper.ts https://arxiv.org/abs/2204.11674"
    );
    console.log("  tsx scripts/fetch-arxiv-paper.ts 2204.11674");
    console.log("");
    console.log("Output: JSON data to stdout");
    process.exit(1);
  }

  const input = args[0];

  try {
    const paperData = await fetchArxivPaper(input);

    // Output as JSON to stdout
    console.log(JSON.stringify(paperData, null, 2));
  } catch (error) {
    console.error(
      `❌ Failed to fetch paper: ${error instanceof Error ? error.message : "Unknown error"}`
    );
    process.exit(1);
  }
}

// Run main function if this script is executed directly
if (require.main === module) {
  main().catch((error) => {
    console.error(
      `❌ Script failed: ${error instanceof Error ? error.message : "Unknown error"}`
    );
    process.exit(1);
  });
}
