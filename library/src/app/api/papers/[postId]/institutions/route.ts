import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db/prisma";
import { BadRequestError } from "@/lib/errors";
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

    // Check if the post has a linked paper with institutions
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

    const hasInstitutions =
      post?.paper?.paperInstitutions && post.paper.paperInstitutions.length > 0;

    return NextResponse.json({ hasInstitutions });
  } catch (error) {
    return handleApiError(error, {
      endpoint: "GET /api/papers/[postId]/institutions",
    });
  }
}
