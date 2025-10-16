import { prisma } from "@/lib/db/prisma";
import { Logger } from "./logging/logger";

/**
 * Service for managing the tag vocabulary from existing papers
 */
export class TagVocabularyService {
  private static cachedTags: string[] | null = null;
  private static cacheTimestamp: number = 0;
  private static readonly CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

  /**
   * Get all unique tags from the database with caching
   */
  static async getAllTags(): Promise<string[]> {
    const now = Date.now();

    // Return cached tags if they're still fresh
    if (
      this.cachedTags &&
      this.cacheTimestamp &&
      now - this.cacheTimestamp < this.CACHE_DURATION
    ) {
      return this.cachedTags;
    }

    Logger.info("🏷️ Refreshing tag vocabulary from database...");

    // Get all papers with tags
    const papers = await prisma.paper.findMany({
      where: {
        NOT: { tags: { equals: [] } },
      },
      select: { tags: true },
    });

    // Extract unique tags
    const tagSet = new Set<string>();
    papers.forEach((paper) => {
      paper.tags.forEach((tag) => {
        if (tag && tag.trim()) {
          tagSet.add(tag.trim());
        }
      });
    });

    const allTags = Array.from(tagSet).sort();

    // Update cache
    this.cachedTags = allTags;
    this.cacheTimestamp = now;

    Logger.info(`✅ Found ${allTags.length} unique tags in vocabulary`);

    return allTags;
  }

  /**
   * Clear the cache (useful when tags are updated)
   */
  static clearCache(): void {
    this.cachedTags = null;
    this.cacheTimestamp = 0;
  }

  /**
   * Get tag vocabulary as a formatted string for LLM prompts
   */
  static async getTagVocabularyForPrompt(): Promise<string> {
    const tags = await this.getAllTags();

    if (tags.length === 0) {
      return "No tags available in the system yet.";
    }

    // Group tags by category for better organization
    const categorizedTags = this.categorizeTags(tags);

    let prompt = "Available tags:\n";

    if (categorizedTags.primary.length > 0) {
      prompt += `\nPrimary categories: ${categorizedTags.primary.join(", ")}\n`;
    }

    if (categorizedTags.other.length > 0) {
      prompt += `\nOther tags: ${categorizedTags.other.join(", ")}\n`;
    }

    return prompt;
  }

  /**
   * Simple categorization of tags for better prompt organization
   */
  private static categorizeTags(tags: string[]): {
    primary: string[];
    other: string[];
  } {
    const primaryCategories = [
      "RL",
      "LLMs",
      "Transformer",
      "Memory",
      "Exploration",
      "Multi-Agency",
      "Optimization",
      "Meta-Learning",
      "Evolution",
      "Neural Architecture Search",
      "Active Inference",
      "Continual Learning",
      "Plasticity",
    ];

    const primary = tags.filter((tag) => primaryCategories.includes(tag));
    const other = tags.filter((tag) => !primaryCategories.includes(tag));

    return { primary, other };
  }
}
