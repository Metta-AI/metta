"use server";

import { prisma } from "@/lib/db/prisma";
import { NotFoundError, BadRequestError } from "@/lib/errors";

export interface PaperDataResult {
  institutions: string[];
  paper: {
    title: string;
    abstract: string | null;
    tags: string[];
    link: string | null;
    source: string | null;
    externalId: string | null;
    stars: number;
    createdAt: Date;
    updatedAt: Date;
    institutions: string[];
  };
}

/**
 * Server action to fetch paper data by post ID
 */
export async function getPaperDataAction(
  postId: string
): Promise<PaperDataResult> {
  if (!postId) {
    throw new BadRequestError("Post ID is required");
  }

  const post = await prisma.post.findUnique({
    where: { id: postId },
    select: {
      paper: {
        select: {
          title: true,
          abstract: true,
          tags: true,
          link: true,
          source: true,
          externalId: true,
          stars: true,
          createdAt: true,
          updatedAt: true,
          paperInstitutions: {
            select: {
              institution: {
                select: {
                  id: true,
                  name: true,
                },
              },
            },
          },
        },
      },
    },
  });

  if (!post?.paper) {
    throw new NotFoundError("Paper", postId);
  }

  const institutions = post.paper.paperInstitutions.map(
    (pi) => pi.institution.name
  );

  return {
    institutions,
    paper: {
      ...post.paper,
      institutions,
    },
  };
}

/**
 * Server action to check if a paper has institutions
 */
export async function checkPaperHasInstitutionsAction(
  postId: string
): Promise<{ hasInstitutions: boolean }> {
  if (!postId) {
    throw new BadRequestError("Post ID is required");
  }

  const post = await prisma.post.findUnique({
    where: { id: postId },
    select: {
      paper: {
        select: {
          paperInstitutions: {
            select: {
              id: true,
            },
          },
        },
      },
    },
  });

  const hasInstitutions = Boolean(
    post?.paper?.paperInstitutions && post.paper.paperInstitutions.length > 0
  );

  return { hasInstitutions };
}
