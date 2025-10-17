import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db/prisma";
import { handleApiError } from "@/lib/api/error-handler";

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const paper = await prisma.paper.findUnique({
      where: { id: params.id },
      include: {
        paperAuthors: {
          include: {
            author: {
              select: {
                id: true,
                name: true,
                institution: true,
              },
            },
          },
        },
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
    });

    if (!paper) {
      return NextResponse.json({ error: "Paper not found" }, { status: 404 });
    }

    // Transform to API format
    const response = {
      id: paper.id,
      title: paper.title,
      abstract: paper.abstract,
      link: paper.link,
      source: paper.source,
      doi: paper.doi,
      arxivUrl: paper.arxivUrl,
      tags: paper.tags || [],
      stars: paper.stars,
      citationCount: paper.citationCount,
      abstractSummary: paper.abstractSummary,
      createdAt: paper.createdAt.toISOString(),
      authors: paper.paperAuthors.map((pa) => ({
        id: pa.author.id,
        name: pa.author.name,
        institution: pa.author.institution,
      })),
      institutions: paper.paperInstitutions.map((pi) => pi.institution.name),
    };

    return NextResponse.json(response);
  } catch (error) {
    return handleApiError(error, { endpoint: `GET /api/papers/${params.id}` });
  }
}
