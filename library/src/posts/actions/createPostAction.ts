"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import {
  processArxivAutoImport,
  detectArxivUrl,
} from "@/lib/arxiv-auto-import";
import { queueArxivInstitutionProcessing } from "@/lib/background-jobs";

const inputSchema = zfd.formData({
  title: zfd.text(z.string().min(1).max(255)),
  content: zfd.text(z.string().optional()),
  postType: zfd.text(
    z.enum(["user-post", "paper-post", "pure-paper"]).optional()
  ),
  paperId: zfd.text(z.string().optional()), // Added support for paperId
  images: zfd.text(z.string().optional()), // JSON string of image URLs
});

export const createPostAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    // Import arXiv paper synchronously for instant paper preview
    let paperId = input.paperId || null;
    let postType = input.postType || "user-post";
    let arxivUrl: string | null = null;

    // Parse images if provided
    let images: string[] = [];
    if (input.images) {
      try {
        images = JSON.parse(input.images);
      } catch (error) {
        console.error("Error parsing images:", error);
      }
    }

    if (input.content && !paperId) {
      // Check for arXiv URL and import paper immediately (fast - no institutions)
      arxivUrl = detectArxivUrl(input.content);
      if (arxivUrl) {
        const importedPaperId = await processArxivAutoImport(input.content);
        if (importedPaperId) {
          paperId = importedPaperId;
          postType = "paper-post"; // Set as paper post immediately
          console.log(`✅ arXiv paper imported synchronously: ${paperId}`);
        }
      }
    }

    // Require posts to have associated papers
    if (!paperId) {
      throw new Error(
        "Posts must include an arXiv paper link. Please include a valid arXiv URL in your post content (e.g., https://arxiv.org/abs/2301.12345)."
      );
    }

    const post = await prisma.post.create({
      data: {
        title: input.title,
        content: input.content || null,
        postType,
        paperId,
        images,
        authorId: session.user.id,
      },
      select: {
        id: true,
      },
    });

    // If we imported a paper, queue institution enhancement in background
    if (paperId && arxivUrl) {
      console.log("🏛️ Queuing institution processing for paper:", paperId);
      // Fire and forget - enhance paper with institutions
      queueArxivInstitutionProcessing(paperId, arxivUrl).catch((error) => {
        console.error("Failed to queue institution processing:", error);
      });
    }

    revalidatePath("/");

    return { id: post.id };
  });
