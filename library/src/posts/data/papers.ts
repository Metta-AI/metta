import { prisma } from "@/lib/db/prisma";
import { auth } from "@/lib/auth";

/**
 * Paper data structure that matches the database schema
 */
export interface Paper {
  id: string;
  title: string;
  abstract: string | null;
  authors?: { id: string; name: string; orcid?: string | null; institution?: string | null }[];
  institutions: string[] | null;
  tags: string[] | null;
  link: string | null;
  source: string | null;
  externalId: string | null;
  stars: number | null;
  starred: boolean | null;
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

    // Fetch all papers with their authors
    const papers = await prisma.paper.findMany({
      include: {
        paperAuthors: {
          include: {
            author: {
              select: {
                id: true,
                name: true,
                orcid: true,
                institution: true
              }
            }
          }
        }
      }
    });
    
    // Fetch all users
    const users = await prisma.user.findMany();
    
    // Fetch all user interactions
    const interactions = await prisma.userPaperInteraction.findMany();

    // If user is logged in, fetch their specific interactions
    let currentUserInteractions: UserInteraction[] = [];
    if (currentUserId) {
      currentUserInteractions = await prisma.userPaperInteraction.findMany({
        where: { userId: currentUserId },
      });
    }

    // Create a map of current user's interactions for quick lookup
    const currentUserInteractionsMap = new Map<string, UserInteraction>();
    currentUserInteractions.forEach(interaction => {
      currentUserInteractionsMap.set(interaction.paperId, interaction);
    });

    // Add current user context to papers and transform author data
    const papersWithContext: PaperWithUserContext[] = papers.map(paper => {
      const userInteraction = currentUserInteractionsMap.get(paper.id);
      return {
        ...paper,
        authors: paper.paperAuthors.map(pa => ({
          id: pa.author.id,
          name: pa.author.name,
          orcid: pa.author.orcid,
          institution: pa.author.institution
        })),
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
    // Fetch all papers with their authors
    const papers = await prisma.paper.findMany({
      include: {
        paperAuthors: {
          include: {
            author: {
              select: {
                id: true,
                name: true,
                orcid: true,
                institution: true
              }
            }
          }
        }
      }
    });
    
    // Fetch all users
    const users = await prisma.user.findMany();
    
    // Fetch all user interactions
    const interactions = await prisma.userPaperInteraction.findMany();
    
    // Transform papers to include author data
    const papersWithAuthors = papers.map(paper => ({
      ...paper,
      authors: paper.paperAuthors.map(pa => ({
        id: pa.author.id,
        name: pa.author.name,
        orcid: pa.author.orcid,
        institution: pa.author.institution
      }))
    }));
    
    return {
      papers: papersWithAuthors,
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
    // Fetch all papers with their authors
    const papers = await prisma.paper.findMany({
      include: {
        paperAuthors: {
          include: {
            author: {
              select: {
                id: true,
                name: true,
                orcid: true,
                institution: true
              }
            }
          }
        }
      }
    });
    
    // Fetch user interactions for the specific user
    const userInteractions = await prisma.userPaperInteraction.findMany({
      where: { userId },
    });
    
    // Transform papers to include author data
    const papersWithAuthors = papers.map(paper => ({
      ...paper,
      authors: paper.paperAuthors.map(pa => ({
        id: pa.author.id,
        name: pa.author.name,
        orcid: pa.author.orcid,
        institution: pa.author.institution
      }))
    }));
    
    return {
      papers: papersWithAuthors,
      userInteractions
    };
  } catch (error) {
    console.error('Error loading papers for user:', error);
    throw new Error('Failed to load papers for user');
  }
} 