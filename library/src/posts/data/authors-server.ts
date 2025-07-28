import { prisma } from "@/lib/db/prisma";

export type AuthorDTO = {
  id: string;
  name: string;
  username: string | null;
  email: string | null;
  avatar: string | null;
  institution: string | null;
  department: string | null;
  title: string | null;
  expertise: string[];
  hIndex: number | null;
  totalCitations: number | null;
  claimed: boolean;
  isFollowing: boolean;
  recentActivity: Date | null;
  orcid: string | null;
  googleScholarId: string | null;
  arxivId: string | null;
  createdAt: Date;
  updatedAt: Date;
  paperCount: number;
  recentPapers: Array<{
    id: string;
    title: string;
    link: string | null;
    createdAt: Date;
    stars: number;
  }>;
};

export async function loadAuthors(): Promise<AuthorDTO[]> {
  const authors = await prisma.author.findMany({
    include: {
      paperAuthors: {
        include: {
          paper: {
            select: {
              id: true,
              title: true,
              link: true,
              createdAt: true,
              stars: true,
            },
          },
        },
        orderBy: {
          paper: {
            createdAt: 'desc',
          },
        },
        take: 5, // Get only the 5 most recent papers for performance
      },
    },
    orderBy: {
      name: 'asc',
    },
  });

  const mappedAuthors = authors.map(author => {
    const recentPapers = author.paperAuthors.map(pa => pa.paper);
    
    return {
      id: author.id,
      name: author.name,
      username: author.username,
      email: author.email,
      avatar: author.avatar,
      institution: author.institution,
      department: author.department,
      title: author.title,
      expertise: author.expertise,
      hIndex: author.hIndex,
      totalCitations: author.totalCitations,
      claimed: author.claimed,
      isFollowing: false, // Default to not following for now
      recentActivity: author.recentActivity,
      orcid: author.orcid,
      googleScholarId: author.googleScholarId,
      arxivId: author.arxivId,
      createdAt: author.createdAt,
      updatedAt: author.updatedAt,
      paperCount: author.paperAuthors.length,
      recentPapers,
    };
  });

  // Add a sample claimed author for demonstration
  const sampleClaimedAuthor: AuthorDTO = {
    id: 'sample-claimed-author',
    name: 'Dr. Sarah Chen',
    username: '@sarahchen',
    email: 'sarah.chen@stanford.edu',
    avatar: null,
    institution: 'Stanford University',
    department: 'Computer Science',
    title: 'Associate Professor',
    expertise: ['Machine Learning', 'Computer Vision', 'Deep Learning', 'AI Safety'],
    hIndex: 42,
    totalCitations: 15420,
    claimed: true,
    isFollowing: false, // Default to not following
    recentActivity: new Date('2024-01-15'),
    orcid: '0000-0001-2345-6789',
    googleScholarId: 'sarahchen',
    arxivId: 'sarahchen',
    createdAt: new Date('2020-01-01'),
    updatedAt: new Date('2024-01-15'),
    paperCount: 15,
    recentPapers: [
      {
        id: 'paper-1',
        title: 'Advances in Multi-Modal Learning for Computer Vision',
        link: 'https://arxiv.org/abs/2024.001',
        createdAt: new Date('2024-01-10'),
        stars: 45,
      },
      {
        id: 'paper-2',
        title: 'Robust Deep Learning for Autonomous Systems',
        link: 'https://arxiv.org/abs/2023.156',
        createdAt: new Date('2023-12-15'),
        stars: 32,
      },
      {
        id: 'paper-3',
        title: 'Interpretable AI: Methods and Applications',
        link: 'https://arxiv.org/abs/2023.089',
        createdAt: new Date('2023-11-20'),
        stars: 28,
      },
    ],
  };

  return [...mappedAuthors, sampleClaimedAuthor];
}

export async function loadAuthor(authorId: string): Promise<AuthorDTO | null> {
  const author = await prisma.author.findUnique({
    where: { id: authorId },
    include: {
      paperAuthors: {
        include: {
          paper: {
            select: {
              id: true,
              title: true,
              link: true,
              createdAt: true,
              stars: true,
              abstract: true,
              institutions: true,
              tags: true,
            },
          },
        },
        orderBy: {
          paper: {
            createdAt: 'desc',
          },
        },
      },
    },
  });

  if (!author) {
    return null;
  }

  const recentPapers = author.paperAuthors.map(pa => ({
    id: pa.paper.id,
    title: pa.paper.title,
    link: pa.paper.link,
    createdAt: pa.paper.createdAt,
    stars: pa.paper.stars,
    abstract: pa.paper.abstract,
    institutions: pa.paper.institutions,
    tags: pa.paper.tags,
  }));

  return {
    id: author.id,
    name: author.name,
    username: author.username,
    email: author.email,
    avatar: author.avatar,
    institution: author.institution,
    department: author.department,
    title: author.title,
    expertise: author.expertise,
    hIndex: author.hIndex,
    totalCitations: author.totalCitations,
    claimed: author.claimed,
    isFollowing: false, // Default to not following for now
    recentActivity: author.recentActivity,
    orcid: author.orcid,
    googleScholarId: author.googleScholarId,
    arxivId: author.arxivId,
    createdAt: author.createdAt,
    updatedAt: author.updatedAt,
    paperCount: author.paperAuthors.length,
    recentPapers,
  };
} 