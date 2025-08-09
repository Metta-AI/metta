#!/usr/bin/env tsx

/**
 * Show Database Contents Script
 * 
 * This script displays the current database contents in a readable format
 */

import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

import { PrismaClient } from '@prisma/client';

// Initialize Prisma client
const prisma = new PrismaClient();

async function showDatabaseContents() {
  console.log('📊 Database Contents\n');

  try {
    // Get all authors with their papers
    const authors = await prisma.author.findMany({
      include: {
        paperAuthors: {
          include: {
            paper: {
              select: { 
                id: true,
                title: true, 
                externalId: true,
                source: true,
                abstract: true
              }
            }
          }
        }
      },
      orderBy: { name: 'asc' }
    });

    // Get all papers with their authors
    const papers = await prisma.paper.findMany({
      include: {
        paperAuthors: {
          include: {
            author: true
          }
        }
      },
      orderBy: { title: 'asc' }
    });

    // Display Authors Table
    console.log('👥 AUTHORS TABLE');
    console.log('='.repeat(80));
    console.log(`${'ID'.padEnd(25)} | ${'Name'.padEnd(25)} | ${'ORCID'.padEnd(20)} | ${'Papers'.padEnd(10)}`);
    console.log('-'.repeat(80));
    
    authors.forEach(author => {
      const id = author.id.substring(0, 8) + '...';
      const name = author.name.padEnd(25);
      const orcid = (author.orcid || 'N/A').padEnd(20);
      const paperCount = author.paperAuthors.length.toString().padEnd(10);
      console.log(`${id} | ${name} | ${orcid} | ${paperCount}`);
    });
    
    console.log(`\nTotal Authors: ${authors.length}\n`);

    // Display Papers Table
    console.log('📄 PAPERS TABLE');
    console.log('='.repeat(120));
    console.log(`${'ID'.padEnd(25)} | ${'Title'.padEnd(50)} | ${'arXiv ID'.padEnd(15)} | ${'Authors'.padEnd(20)}`);
    console.log('-'.repeat(120));
    
    papers.forEach(paper => {
      const id = paper.id.substring(0, 8) + '...';
      const title = paper.title.length > 47 ? paper.title.substring(0, 47) + '...' : paper.title;
      const arxivId = (paper.externalId || 'N/A').padEnd(15);
      const authorNames = paper.paperAuthors.map(pa => pa.author.name).join(', ');
      const authors = authorNames.length > 17 ? authorNames.substring(0, 17) + '...' : authorNames;
      
      console.log(`${id} | ${title.padEnd(50)} | ${arxivId} | ${authors.padEnd(20)}`);
    });
    
    console.log(`\nTotal Papers: ${papers.length}\n`);

    // Display Paper-Author Relationships
    console.log('🔗 PAPER-AUTHOR RELATIONSHIPS');
    console.log('='.repeat(100));
    console.log(`${'Paper ID'.padEnd(25)} | ${'Author ID'.padEnd(25)} | ${'Paper Title'.padEnd(40)}`);
    console.log('-'.repeat(100));
    
    const relationships = await prisma.paperAuthor.findMany({
      include: {
        paper: { select: { title: true } },
        author: { select: { name: true } }
      }
    });
    
    relationships.forEach(rel => {
      const paperId = rel.paperId.substring(0, 8) + '...';
      const authorId = rel.authorId.substring(0, 8) + '...';
      const title = rel.paper.title.length > 37 ? rel.paper.title.substring(0, 37) + '...' : rel.paper.title;
      
      console.log(`${paperId} | ${authorId} | ${title.padEnd(40)}`);
    });
    
    console.log(`\nTotal Relationships: ${relationships.length}\n`);

    // Summary Statistics
    console.log('📈 SUMMARY STATISTICS');
    console.log('='.repeat(50));
    console.log(`Total Authors: ${authors.length}`);
    console.log(`Total Papers: ${papers.length}`);
    console.log(`Total Relationships: ${relationships.length}`);
    console.log(`Papers with Authors: ${papers.filter(p => p.paperAuthors.length > 0).length}`);
    console.log(`Authors with Papers: ${authors.filter(a => a.paperAuthors.length > 0).length}`);
    console.log(`Authors with ORCID: ${authors.filter(a => a.orcid).length}`);
    console.log(`Authors with Institution: ${authors.filter(a => a.institution).length}`);
    console.log(`Authors with Email: ${authors.filter(a => a.email).length}`);

    // Show sample data in detail
    if (authors.length > 0) {
      console.log('\n🔍 SAMPLE AUTHOR DETAIL');
      console.log('='.repeat(50));
      const sampleAuthor = authors[0];
      console.log(`Name: ${sampleAuthor.name}`);
      console.log(`ID: ${sampleAuthor.id}`);
      console.log(`ORCID: ${sampleAuthor.orcid || 'N/A'}`);
      console.log(`Institution: ${sampleAuthor.institution || 'N/A'}`);
      console.log(`Email: ${sampleAuthor.email || 'N/A'}`);
      console.log(`Created: ${sampleAuthor.createdAt.toISOString()}`);
      console.log(`Papers: ${sampleAuthor.paperAuthors.map(pa => pa.paper.title).join(', ')}`);
    }

    if (papers.length > 0) {
      console.log('\n🔍 SAMPLE PAPER DETAIL');
      console.log('='.repeat(50));
      const samplePaper = papers[0];
      console.log(`Title: ${samplePaper.title}`);
      console.log(`ID: ${samplePaper.id}`);
      console.log(`arXiv ID: ${samplePaper.externalId || 'N/A'}`);
      console.log(`Source: ${samplePaper.source || 'N/A'}`);
      console.log(`Abstract: ${samplePaper.abstract ? samplePaper.abstract.substring(0, 100) + '...' : 'N/A'}`);
      console.log(`Authors: ${samplePaper.paperAuthors.map(pa => pa.author.name).join(', ')}`);
    }

  } catch (error) {
    console.error('❌ Error displaying database contents:', error);
  }
}

showDatabaseContents().catch(console.error); 