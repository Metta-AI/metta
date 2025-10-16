#!/usr/bin/env tsx

/**
 * Asana Paper Collection Script
 *
 * This script pulls paper collections from an Asana project and imports them into the library database.
 * Each task in the Asana project represents a research paper with metadata in the task title, description, and custom fields.
 */

import { PrismaClient } from "@prisma/client";
import * as dotenv from "dotenv";
import { fetchArxivPaper } from "./fetch-arxiv-paper";

// Load environment variables (load .env first, then .env.local for overrides)
dotenv.config({ path: ".env", quiet: true });
dotenv.config({ path: ".env.local", quiet: true });

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
    multi_enum_values?: Array<{
      name: string;
    }>;
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
  authors: string[];
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
    throw new Error(
      "ASANA_API_KEY or ASANA_TOKEN environment variable is required"
    );
  }

  if (!config.projectId) {
    throw new Error("ASANA_PAPERS_PROJECT_ID environment variable is required");
  }

  console.log(`📋 Fetching tasks from Asana project: ${config.projectId}`);

  const tasks: AsanaTask[] = [];
  let offset: string | undefined;

  do {
    const url = `https://app.asana.com/api/1.0/projects/${config.projectId}/tasks`;
    const params = new URLSearchParams({
      opt_fields:
        "gid,name,notes,completed,permalink_url,custom_fields.gid,custom_fields.name,custom_fields.text_value,custom_fields.enum_value.name,custom_fields.multi_enum_values.name,created_at,modified_at",
      limit: "100",
    });

    if (offset) {
      params.append("offset", offset);
    }

    const response = await fetch(`${url}?${params}`, {
      headers: {
        Authorization: `Bearer ${config.asanaToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(
        `Asana API Error: ${response.status} - ${await response.text()}`
      );
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
async function extractPaperData(task: AsanaTask): Promise<PaperData | null> {
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
  let authorsText: string | undefined;

  // Debug: Log all custom fields (disabled by default, uncomment to debug)
  // if (task.custom_fields && task.custom_fields.length > 0) {
  //   console.log(`\n🔍 Custom fields for "${title}":`);
  //   task.custom_fields.forEach((field) => {
  //     const value = field.text_value || field.enum_value?.name;
  //     console.log(`   - ${field.name}: ${value || "(empty)"}`);
  //   });
  // }

  for (const field of task.custom_fields || []) {
    const value = field.text_value || field.enum_value?.name;
    if (!value) continue;

    if (
      field.gid === config.paperLinkFieldId ||
      field.name.toLowerCase().includes("link") ||
      field.name.toLowerCase().includes("url")
    ) {
      paperLink = value;
    } else if (
      field.gid === config.arxivIdFieldId ||
      field.name.toLowerCase().includes("arxiv")
    ) {
      arxivId = value;
    } else if (
      field.gid === config.abstractFieldId ||
      field.name.toLowerCase().includes("abstract")
    ) {
      customAbstract = value;
    } else if (
      field.name.toLowerCase().includes("author") ||
      field.name.toLowerCase().includes("researcher")
    ) {
      authorsText = value;
      console.log(`   ✅ Found authors field: "${field.name}" = "${value}"`);
    }
  }

  // Use custom abstract if available and longer than notes
  if (
    customAbstract &&
    (!abstract || customAbstract.length > abstract.length)
  ) {
    abstract = customAbstract;
  }

  // Determine source and external ID
  let source = "asana";
  let externalId = task.gid;
  let link = paperLink;

  // Check if we have an arXiv paper
  if (arxivId || (paperLink && paperLink.includes("arxiv.org"))) {
    source = "arxiv";
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
  } else if (paperLink && paperLink.includes("biorxiv.org")) {
    source = "biorxiv";
    link = paperLink;
  } else if (paperLink) {
    source = "external";
    link = paperLink;
  }

  // Extract tags from "Category" custom field
  let tags: string[] = [];

  for (const field of task.custom_fields || []) {
    if (field.name.toLowerCase().includes("category")) {
      // Handle multi-enum values (most common for categories)
      if (field.multi_enum_values && Array.isArray(field.multi_enum_values)) {
        tags = field.multi_enum_values
          .map((val: any) => val.name)
          .filter(Boolean);
      }
      // Handle single enum value
      else if (field.enum_value?.name) {
        tags = [field.enum_value.name];
      }
      // Handle text values (comma-separated)
      else if (field.text_value) {
        tags = field.text_value
          .split(",")
          .map((tag) => tag.trim())
          .filter(Boolean);
      }
      break; // Found category field, stop looking
    }
  }

  // Parse authors from text field (comma-separated, semicolon-separated, or newline-separated)
  let authors: string[] = [];
  if (authorsText) {
    // Split by common separators and clean up
    const authorList = authorsText
      .split(/[,;\n]/)
      .map((author) => author.trim())
      .filter((author) => author.length > 0);
    authors.push(...authorList);
  }

  // If no authors found and this is an arXiv paper, fetch from arXiv API
  if (authors.length === 0 && source === "arxiv" && externalId) {
    try {
      console.log(`   🔍 Fetching authors from arXiv for: ${title}`);
      const arxivData = await fetchArxivPaper(externalId);
      authors = arxivData.authors;
      // Also update abstract if not present
      if (!abstract && arxivData.abstract) {
        abstract = arxivData.abstract;
      }
      console.log(`   ✅ Found ${authors.length} authors from arXiv`);
    } catch (error) {
      console.log(
        `   ⚠️ Could not fetch arXiv data:`,
        error instanceof Error ? error.message : String(error)
      );
    }
  }

  // Debug: Log tasks with tags and authors
  if (tags.length > 0) {
    console.log(`📋 Task "${title}" has category tags: ${tags.join(", ")}`);
  }
  if (authors.length > 0) {
    console.log(`👥 Task "${title}" has authors: ${authors.join(", ")}`);
  }

  return {
    title,
    abstract,
    link,
    source,
    externalId,
    tags,
    authors,
    asanaId: task.gid,
    asanaUrl: task.permalink_url,
  };
}

/**
 * Find or create an author by name
 */
async function findOrCreateAuthor(name: string): Promise<string> {
  const trimmedName = name.trim();

  // Check if author already exists
  let author = await prisma.author.findUnique({
    where: { name: trimmedName },
  });

  // Create author if not found
  if (!author) {
    author = await prisma.author.create({
      data: {
        name: trimmedName,
      },
    });
    console.log(`      👤 Created new author: ${trimmedName}`);
  }

  return author.id;
}

/**
 * Link authors to a paper
 */
async function linkAuthorsToPaper(
  paperId: string,
  authorNames: string[]
): Promise<void> {
  if (authorNames.length === 0) return;

  for (const authorName of authorNames) {
    try {
      const authorId = await findOrCreateAuthor(authorName);

      // Create paper-author relationship (skip if already exists)
      await prisma.paperAuthor.upsert({
        where: {
          paperId_authorId: {
            paperId,
            authorId,
          },
        },
        create: {
          paperId,
          authorId,
        },
        update: {}, // No updates needed if already exists
      });
    } catch (error) {
      console.error(`      ⚠️ Error linking author "${authorName}":`, error);
    }
  }
}

/**
 * Import papers into the database
 */
async function importPapers(papers: PaperData[]): Promise<void> {
  console.log(`📚 Importing ${papers.length} papers into database...`);

  let imported = 0;
  let skipped = 0;
  let errors = 0;
  let authorsCreated = 0;

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
        select: {
          id: true,
          title: true,
          tags: true,
          paperAuthors: {
            include: {
              author: true,
            },
          },
        },
      });

      if (existing) {
        // Update existing paper with tags if it doesn't have any or if tags have changed
        if (
          !existing.tags ||
          existing.tags.length === 0 ||
          JSON.stringify(existing.tags.sort()) !==
            JSON.stringify(paper.tags.sort())
        ) {
          await prisma.paper.update({
            where: { id: existing.id },
            data: { tags: paper.tags },
          });

          console.log(`   🔄 Updated tags for existing paper: ${paper.title}`);
          console.log(`      New tags: [${paper.tags.join(", ")}]`);
        }

        // Link authors if they're missing
        if (paper.authors.length > 0 && existing.paperAuthors.length === 0) {
          console.log(`   👥 Adding authors to existing paper: ${paper.title}`);
          await linkAuthorsToPaper(existing.id, paper.authors);
        } else {
          console.log(`   ⏭️  Skipping existing paper: ${paper.title}`);
        }

        skipped++;
        continue;
      }

      // Create the paper
      const newPaper = await prisma.paper.create({
        data: {
          title: paper.title,
          abstract: paper.abstract,
          link: paper.link,
          source: paper.source,
          externalId: paper.externalId,
          tags: paper.tags,
        },
      });

      // Link authors to the new paper
      if (paper.authors.length > 0) {
        await linkAuthorsToPaper(newPaper.id, paper.authors);
        authorsCreated += paper.authors.length;
      }

      console.log(`   ✅ Imported: ${paper.title}`);
      if (paper.authors.length > 0) {
        console.log(`      👥 Linked ${paper.authors.length} author(s)`);
      }
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
  console.log(`   - Author relationships created: ${authorsCreated}`);
}

/**
 * Main function
 */
async function main() {
  console.log("🚀 Starting Asana paper collection import...\n");

  try {
    // Fetch tasks from Asana
    const tasks = await fetchAsanaTasks();
    console.log(`   Found ${tasks.length} tasks\n`);

    // Extract paper data
    console.log("🔍 Extracting paper data from tasks...");
    const papers: PaperData[] = [];

    for (const task of tasks) {
      const paper = await extractPaperData(task);
      if (paper) {
        papers.push(paper);
      }
    }

    console.log(`   Extracted ${papers.length} valid papers\n`);

    if (papers.length === 0) {
      console.log("ℹ️  No papers to import");
      return;
    }

    // Import into database
    await importPapers(papers);

    console.log("\n🎉 Asana paper collection import completed successfully!");
  } catch (error) {
    console.error("❌ Error during import:", error);
    if (error instanceof Error) {
      console.error("Error details:", error.message);
    }
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

// Run the script
main().catch(console.error);
