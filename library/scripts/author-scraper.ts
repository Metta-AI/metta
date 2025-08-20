#!/usr/bin/env tsx

/**
 * arXiv Author Scraper Script
 * 
 * This script fetches papers by author from arXiv using the au: field prefix.
 * It can be used both as a command-line tool and imported as a module.
 * 
 * Usage as CLI:
 *   tsx scripts/author-scraper.ts "Ada Lovelace"
 *   tsx scripts/author-scraper.ts "Ada Lovelace" --max-results 50
 * 
 * Usage as module:
 *   import { fetchPapersByAuthor } from './scripts/author-scraper';
 *   const papers = await fetchPapersByAuthor('Ada Lovelace', 100);
 */

import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

// XML parsing library for arXiv RSS feed
import { XMLParser } from 'fast-xml-parser';

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
 * Interface for the raw arXiv API response with multiple entries
 */
interface ArxivApiResponse {
  feed: {
    'opensearch:totalResults'?: { _: string };
    entry: {
      id: string;
      title: string;
      summary: string;
      published: string;
      updated: string;
      author: { name: string } | Array<{ name: string }>;
      category: { _: string; term: string } | Array<{ _: string; term: string }>;
      link: { href: string; title?: string; type?: string } | Array<{ href: string; title?: string; type?: string }>;
      'arxiv:doi'?: { _: string };
      'arxiv:journal_ref'?: { _: string };
      'arxiv:primary_category'?: { term: string };
    } | Array<{
      id: string;
      title: string;
      summary: string;
      published: string;
      updated: string;
      author: { name: string } | Array<{ name: string }>;
      category: { _: string; term: string } | Array<{ _: string; term: string }>;
      link: { href: string; title?: string; type?: string } | Array<{ href: string; title?: string; type?: string }>;
      'arxiv:doi'?: { _: string };
      'arxiv:journal_ref'?: { _: string };
      'arxiv:primary_category'?: { term: string };
    }>;
  };
}

/**
 * Interface for the response containing papers and metadata
 */
export interface AuthorSearchResult {
  papers: ArxivPaperData[];
  totalResults: number;
  authorName: string;
  maxResults: number;
}

/**
 * Fetches papers by author from arXiv API
 * 
 * @param authorName - The author name to search for (e.g., "Ada Lovelace")
 * @param maxResults - Maximum number of results to return (default: 100, max: 2000)
 * @returns Promise resolving to structured paper data and metadata
 */
export async function fetchPapersByAuthor(
  authorName: string, 
  maxResults: number = 100
): Promise<AuthorSearchResult> {
  try {
    // Validate and sanitize inputs
    if (!authorName || authorName.trim().length === 0) {
      throw new Error('Author name cannot be empty');
    }
    
    if (maxResults < 1 || maxResults > 2000) {
      throw new Error('maxResults must be between 1 and 2000');
    }
    
    const cleanAuthorName = authorName.trim();
    
    // Construct the arXiv API URL with author search
    const searchQuery = `au:"${encodeURIComponent(cleanAuthorName)}"`;
    const apiUrl = `http://export.arxiv.org/api/query?search_query=${searchQuery}&max_results=${maxResults}`;
    
    console.log(`🔍 Fetching papers for author: "${cleanAuthorName}" (max: ${maxResults})...`);
    
    // Fetch data from arXiv API
    const response = await fetch(apiUrl);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const xmlData = await response.text();
    
    // Parse XML response
    const parser = new XMLParser({
      ignoreAttributes: false,
      attributeNamePrefix: '',
      textNodeName: '_',
      parseAttributeValue: false
    });
    
    const parsedData = parser.parse(xmlData) as ArxivApiResponse;
    
    // Check if we got a valid response
    if (!parsedData.feed?.entry) {
      console.log(`ℹ️  No papers found for author: "${cleanAuthorName}"`);
      return {
        papers: [],
        totalResults: 0,
        authorName: cleanAuthorName,
        maxResults
      };
    }
    
    // Extract total results count
    const totalResults = parseInt(parsedData.feed['opensearch:totalResults']?._ || '0', 10);
    
    // Handle single entry vs multiple entries
    const entries = Array.isArray(parsedData.feed.entry) 
      ? parsedData.feed.entry 
      : [parsedData.feed.entry];
    
    // Process each entry into structured paper data
    const papers: ArxivPaperData[] = entries.map(entry => {
      // Extract authors
      const authors = Array.isArray(entry.author) 
        ? entry.author.map(author => author.name)
        : [entry.author.name];
      
      // Extract categories
      const categories = Array.isArray(entry.category)
        ? entry.category.map(cat => cat.term)
        : [entry.category.term];
      
      // Extract links
      const links = Array.isArray(entry.link) ? entry.link : [entry.link];
      const arxivUrl = links.find(link => link.title === 'abs')?.href || '';
      const pdfUrl = links.find(link => link.type === 'application/pdf')?.href || '';
      
      // Extract DOI and journal reference if available
      const doi = entry['arxiv:doi']?._;
      const journalRef = entry['arxiv:journal_ref']?._;
      const primaryCategory = entry['arxiv:primary_category']?.term || categories[0];
      
      // Extract arXiv ID from the entry ID
      const arxivId = entry.id.replace('http://arxiv.org/abs/', '');
      
      return {
        id: arxivId,
        title: entry.title.replace(/\s+/g, ' ').trim(), // Clean up whitespace
        abstract: entry.summary.replace(/\s+/g, ' ').trim(), // Clean up whitespace
        authors,
        categories,
        publishedDate: entry.published,
        updatedDate: entry.updated,
        doi,
        journalRef,
        primaryCategory,
        arxivUrl,
        pdfUrl,
        summary: entry.summary.replace(/\s+/g, ' ').trim() // Alias for abstract
      };
    });
    
    console.log(`✅ Successfully fetched ${papers.length} papers for "${cleanAuthorName}" (total available: ${totalResults})`);
    
    return {
      papers,
      totalResults,
      authorName: cleanAuthorName,
      maxResults
    };
    
  } catch (error) {
    console.error(`❌ Error fetching papers for author: ${error instanceof Error ? error.message : 'Unknown error'}`);
    throw error;
  }
}

/**
 * Main function for command-line usage
 */
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('📚 arXiv Author Scraper');
    console.log('');
    console.log('Usage:');
    console.log('  tsx scripts/author-scraper.ts "Author Name" [--max-results <number>]');
    console.log('');
    console.log('Examples:');
    console.log('  tsx scripts/author-scraper.ts "Ada Lovelace"');
    console.log('  tsx scripts/author-scraper.ts "Ada Lovelace" --max-results 50');
    console.log('');
    console.log('Output: JSON data to stdout');
    process.exit(1);
  }
  
  const authorName = args[0];
  let maxResults = 100; // Default value
  
  // Parse optional max-results argument
  const maxResultsIndex = args.indexOf('--max-results');
  if (maxResultsIndex !== -1 && maxResultsIndex + 1 < args.length) {
    const maxResultsValue = parseInt(args[maxResultsIndex + 1], 10);
    if (!isNaN(maxResultsValue) && maxResultsValue > 0 && maxResultsValue <= 2000) {
      maxResults = maxResultsValue;
    } else {
      console.error('❌ Invalid max-results value. Must be between 1 and 2000.');
      process.exit(1);
    }
  }
  
  try {
    const result = await fetchPapersByAuthor(authorName, maxResults);
    
    // Output as JSON to stdout
    console.log(JSON.stringify(result, null, 2));
    
  } catch (error) {
    console.error(`❌ Failed to fetch papers: ${error instanceof Error ? error.message : 'Unknown error'}`);
    process.exit(1);
  }
}

// Run main function if this script is executed directly
if (require.main === module) {
  main().catch(error => {
    console.error(`❌ Script failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    process.exit(1);
  });
} 