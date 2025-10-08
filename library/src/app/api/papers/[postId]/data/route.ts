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
      return NextResponse.json(
        { error: "Paper not found for this post" },
        { status: 404 }
      );
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
    console.error("Error fetching paper data:", error);
    return NextResponse.json(
      { error: "Failed to fetch paper data" },
      { status: 500 }
    );
  }
}
