import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db/prisma";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ postId: string }> }
) {
  try {
    const { postId } = await params;

    if (!postId) {
      return NextResponse.json(
        { error: "Post ID is required" },
        { status: 400 }
      );
    }

    // Check if the post has a linked paper with institutions
    const post = await prisma.post.findUnique({
      where: { id: postId },
      select: {
        paper: {
          select: { institutions: true },
        },
      },
    });

    const hasInstitutions =
      post?.paper?.institutions && post.paper.institutions.length > 0;

    return NextResponse.json({ hasInstitutions });
  } catch (error) {
    console.error("Error checking institution status:", error);
    return NextResponse.json(
      { error: "Failed to check institution status" },
      { status: 500 }
    );
  }
}
