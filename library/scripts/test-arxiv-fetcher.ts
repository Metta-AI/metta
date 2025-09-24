#!/usr/bin/env tsx

/**
 * Test script demonstrating how to use the arXiv fetcher as a module
 */

import { fetchArxivPaper, ArxivPaperData } from './fetch-arxiv-paper';

async function testArxivFetcher() {
  console.log('🧪 Testing arXiv Paper Fetcher Module\n');
  
  const testPapers = [
    'https://arxiv.org/abs/2204.11674',
    'https://arxiv.org/abs/1706.03762', // Attention is All You Need
    'https://arxiv.org/abs/1810.04805'  // BERT
  ];
  
  for (const paperUrl of testPapers) {
    try {
      console.log(`📄 Testing: ${paperUrl}`);
      const paperData: ArxivPaperData = await fetchArxivPaper(paperUrl);
      
      console.log(`   Title: ${paperData.title}`);
      console.log(`   Authors: ${paperData.authors.join(', ')}`);
      console.log(`   Categories: ${paperData.categories.join(', ')}`);
      console.log(`   Published: ${paperData.publishedDate}`);
      console.log(`   Abstract length: ${paperData.abstract.length} characters`);
      console.log('');
      
    } catch (error) {
      console.error(`   ❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}\n`);
    }
  }
  
  console.log('✅ Module test completed!');
}

// Run the test
testArxivFetcher().catch(error => {
  console.error(`❌ Test failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  process.exit(1);
}); 