import { NextRequest, NextResponse } from "next/server";

import { prisma } from "@/lib/db/prisma";
import { BadRequestError, NotFoundError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ postId: string }> }
) {
  try {
    const { postId } = await params;

    if (!postId) {
      throw new BadRequestError("Post ID is required");
    }

    // Get the paper data for this post
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

    return NextResponse.json({
      institutions,
      paper: {
        ...post.paper,
        institutions,
      },
    });
  } catch (error) {
    return handleApiError(error, { endpoint: "GET /api/papers/[postId]/data" });
  }
}
