#!/usr/bin/env tsx

/**
 * Asana Paper Collection Script
 *
 * This script pulls paper collections from an Asana project and imports them into the library database.
 * Each task in the Asana project represents a research paper with metadata in the task title, description, and custom fields.
 */

import { PrismaClient } from '@prisma/client';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: '.env.local' });

// Initialize Prisma client
const prisma = new PrismaClient();

interface AsanaTask {
  gid: string;
  name: string;
  notes: string;
  completed: boolean;
  permalink_url: string;
  custom_fields: Array<{
    gid: string;
    name: string;
    text_value?: string;
    enum_value?: {
      name: string;
    };
  }>;
  tags: Array<{
    name: string;
  }>;
  created_at: string;
  modified_at: string;
}

interface PaperData {
  title: string;
  abstract?: string;
  link?: string;
  source: string;
  externalId?: string;
  tags: string[];
  asanaId: string;
  asanaUrl: string;
}

/**
 * Configuration from environment variables
 */
const config = {
  asanaToken: process.env.ASANA_API_KEY || process.env.ASANA_TOKEN,
  projectId: process.env.ASANA_PAPERS_PROJECT_ID,
  workspaceId: process.env.ASANA_WORKSPACE_ID,
  // Custom field IDs (these will need to be configured for your Asana setup)
  paperLinkFieldId: process.env.ASANA_PAPER_LINK_FIELD_ID,
  arxivIdFieldId: process.env.ASANA_ARXIV_ID_FIELD_ID,
  abstractFieldId: process.env.ASANA_ABSTRACT_FIELD_ID,
};

/**
 * Fetch all tasks from the Asana papers project
 */
async function fetchAsanaTasks(): Promise<AsanaTask[]> {
  if (!config.asanaToken) {
    throw new Error('ASANA_API_KEY or ASANA_TOKEN environment variable is required');
  }

  if (!config.projectId) {
    throw new Error('ASANA_PAPERS_PROJECT_ID environment variable is required');
  }

  console.log(`📋 Fetching tasks from Asana project: ${config.projectId}`);

  const tasks: AsanaTask[] = [];
  let offset: string | undefined;

  do {
    const url = `https://app.asana.com/api/1.0/projects/${config.projectId}/tasks`;
    const params = new URLSearchParams({
      opt_fields: 'gid,name,notes,completed,permalink_url,custom_fields.gid,custom_fields.name,custom_fields.text_value,custom_fields.enum_value.name,tags.name,created_at,modified_at',
      limit: '100',
    });

    if (offset) {
      params.append('offset', offset);
    }

    const response = await fetch(`${url}?${params}`, {
      headers: {
        'Authorization': `Bearer ${config.asanaToken}`,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Asana API Error: ${response.status} - ${await response.text()}`);
    }

    const data = await response.json();
    tasks.push(...data.data);

    offset = data.next_page?.offset;
    console.log(`   Fetched ${data.data.length} tasks (${tasks.length} total)`);
  } while (offset);

  return tasks;
}

/**
 * Extract paper data from an Asana task
 */
function extractPaperData(task: AsanaTask): PaperData | null {
  const title = task.name.trim();

  // Skip tasks that don't look like papers
  if (!title || task.completed) {
    return null;
  }

  // Extract abstract from notes
  let abstract = task.notes?.trim() || undefined;
  if (abstract && abstract.length < 50) {
    // If notes are too short, probably not an abstract
    abstract = undefined;
  }

  // Extract custom field values
  let paperLink: string | undefined;
  let arxivId: string | undefined;
  let customAbstract: string | undefined;

  for (const field of task.custom_fields || []) {
    const value = field.text_value || field.enum_value?.name;
    if (!value) continue;

    if (field.gid === config.paperLinkFieldId || field.name.toLowerCase().includes('link') || field.name.toLowerCase().includes('url')) {
      paperLink = value;
    } else if (field.gid === config.arxivIdFieldId || field.name.toLowerCase().includes('arxiv')) {
      arxivId = value;
    } else if (field.gid === config.abstractFieldId || field.name.toLowerCase().includes('abstract')) {
      customAbstract = value;
    }
  }

  // Use custom abstract if available and longer than notes
  if (customAbstract && (!abstract || customAbstract.length > abstract.length)) {
    abstract = customAbstract;
  }

  // Determine source and external ID
  let source = 'asana';
  let externalId = task.gid;
  let link = paperLink;

  // Check if we have an arXiv paper
  if (arxivId || (paperLink && paperLink.includes('arxiv.org'))) {
    source = 'arxiv';
    if (arxivId) {
      externalId = arxivId;
      if (!link) {
        link = `https://arxiv.org/abs/${arxivId}`;
      }
    } else if (paperLink) {
      const arxivMatch = paperLink.match(/arxiv\.org\/abs\/(\d+\.\d+)/);
      if (arxivMatch) {
        externalId = arxivMatch[1];
      }
      link = paperLink;
    }
  } else if (paperLink && paperLink.includes('biorxiv.org')) {
    source = 'biorxiv';
    link = paperLink;
  } else if (paperLink) {
    source = 'external';
    link = paperLink;
  }

  // Extract tags
  const tags = (task.tags || []).map(tag => tag.name).filter(Boolean);

  return {
    title,
    abstract,
    link,
    source,
    externalId,
    tags,
    asanaId: task.gid,
    asanaUrl: task.permalink_url,
  };
}

/**
 * Import papers into the database
 */
async function importPapers(papers: PaperData[]): Promise<void> {
  console.log(`📚 Importing ${papers.length} papers into database...`);

  let imported = 0;
  let skipped = 0;
  let errors = 0;

  for (const paper of papers) {
    try {
      // Check if paper already exists (by external ID or Asana ID)
      const existing = await prisma.paper.findFirst({
        where: {
          OR: [
            { externalId: paper.externalId },
            { link: paper.link },
            { title: paper.title },
          ],
        },
      });

      if (existing) {
        console.log(`   ⏭️  Skipping existing paper: ${paper.title}`);
        skipped++;
        continue;
      }

      // Create the paper
      await prisma.paper.create({
        data: {
          title: paper.title,
          abstract: paper.abstract,
          link: paper.link,
          source: paper.source,
          externalId: paper.externalId,
          tags: paper.tags,
          institutions: [], // No institution data from Asana for now
        },
      });

      console.log(`   ✅ Imported: ${paper.title}`);
      imported++;

    } catch (error) {
      console.error(`   ❌ Error importing "${paper.title}":`, error);
      errors++;
    }
  }

  console.log(`\n📊 Import Summary:`);
  console.log(`   - Imported: ${imported}`);
  console.log(`   - Skipped: ${skipped}`);
  console.log(`   - Errors: ${errors}`);
}

/**
 * Main function
 */
async function main() {
  console.log('🚀 Starting Asana paper collection import...\n');

  try {
    // Fetch tasks from Asana
    const tasks = await fetchAsanaTasks();
    console.log(`   Found ${tasks.length} tasks\n`);

    // Extract paper data
    console.log('🔍 Extracting paper data from tasks...');
    const papers = tasks
      .map(extractPaperData)
      .filter((paper): paper is PaperData => paper !== null);

    console.log(`   Extracted ${papers.length} valid papers\n`);

    if (papers.length === 0) {
      console.log('ℹ️  No papers to import');
      return;
    }

    // Import into database
    await importPapers(papers);

    console.log('\n🎉 Asana paper collection import completed successfully!');

  } catch (error) {
    console.error('❌ Error during import:', error);
    if (error instanceof Error) {
      console.error('Error details:', error.message);
    }
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

// Run the script
main().catch(console.error);
