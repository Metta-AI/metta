import { db } from "@/lib/db";
import { papersTable, userPaperInteractionsTable } from "@/lib/db/schema/paper";
import { usersTable } from "@/lib/db/schema/auth";
import { eq, and } from "drizzle-orm";
import { auth } from "@/lib/auth";

/**
 * Paper data structure that matches the database schema
 */
export interface Paper {
  id: string;
  title: string;
  abstract: string | null;
  authors: string[] | null;
  institutions: string[] | null;
  tags: string[] | null;
  link: string | null;
  source: string | null;
  externalId: string | null;
  stars: number | null;
  starred: boolean | null;
  pdfS3Url: string | null;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * User interaction data structure
 */
export interface UserInteraction {
  userId: string;
  paperId: string;
  starred: boolean | null;
  readAt: Date | null;
  queued: boolean | null;
  notes: string | null;
}

/**
 * User data structure
 */
export interface User {
  id: string;
  name: string | null;
  email: string | null;
  image: string | null;
}

/**
 * Paper with current user context
 */
export interface PaperWithUserContext extends Paper {
  isStarredByCurrentUser: boolean;
  isQueuedByCurrentUser: boolean;
}

/**
 * Load all papers from the database with their user interactions and current user context
 * 
 * This function fetches papers and their associated user interactions,
 * and determines the current user's interaction status for each paper.
 */
export async function loadPapersWithUserContext(): Promise<{
  papers: PaperWithUserContext[];
  users: User[];
  interactions: UserInteraction[];
}> {
  try {
    // Get current user session
    const session = await auth();
    const currentUserId = session?.user?.id;

    // Fetch all papers
    const papers = await db.select().from(papersTable);
    
    // Fetch all users
    const users = await db.select().from(usersTable);
    
    // Fetch all user interactions
    const interactions = await db.select().from(userPaperInteractionsTable);

    // If user is logged in, fetch their specific interactions
    let currentUserInteractions: UserInteraction[] = [];
    if (currentUserId) {
      currentUserInteractions = await db
        .select()
        .from(userPaperInteractionsTable)
        .where(eq(userPaperInteractionsTable.userId, currentUserId));
    }

    // Create a map of current user's interactions for quick lookup
    const currentUserInteractionsMap = new Map<string, UserInteraction>();
    currentUserInteractions.forEach(interaction => {
      currentUserInteractionsMap.set(interaction.paperId, interaction);
    });

    // Add current user context to papers
    const papersWithContext: PaperWithUserContext[] = papers.map(paper => {
      const userInteraction = currentUserInteractionsMap.get(paper.id);
      return {
        ...paper,
        isStarredByCurrentUser: userInteraction?.starred || false,
        isQueuedByCurrentUser: userInteraction?.queued || false,
      };
    });
    
    return {
      papers: papersWithContext,
      users,
      interactions
    };
  } catch (error) {
    console.error('Error loading papers with user context:', error);
    throw new Error('Failed to load papers from database');
  }
}

/**
 * Load all papers from the database with their user interactions
 * 
 * This function fetches papers and their associated user interactions,
 * transforming the data to match the expected format for the papers view.
 */
export async function loadPapers(): Promise<{
  papers: Paper[];
  users: User[];
  interactions: UserInteraction[];
}> {
  try {
    // Fetch all papers
    const papers = await db.select().from(papersTable);
    
    // Fetch all users
    const users = await db.select().from(usersTable);
    
    // Fetch all user interactions
    const interactions = await db.select().from(userPaperInteractionsTable);
    
    return {
      papers,
      users,
      interactions
    };
  } catch (error) {
    console.error('Error loading papers:', error);
    throw new Error('Failed to load papers from database');
  }
}

/**
 * Load papers with user interaction data for a specific user
 * 
 * @param userId - The ID of the user to load interactions for
 */
export async function loadPapersForUser(userId: string): Promise<{
  papers: Paper[];
  userInteractions: UserInteraction[];
}> {
  try {
    // Fetch all papers
    const papers = await db.select().from(papersTable);
    
    // Fetch user interactions for the specific user
    const userInteractions = await db
      .select()
      .from(userPaperInteractionsTable)
      .where(eq(userPaperInteractionsTable.userId, userId));
    
    return {
      papers,
      userInteractions
    };
  } catch (error) {
    console.error('Error loading papers for user:', error);
    throw new Error('Failed to load papers for user');
  }
} 