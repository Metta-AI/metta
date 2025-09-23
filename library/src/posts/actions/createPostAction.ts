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
import { parseMentions } from "@/lib/mentions";
import {
  resolveMentions,
  extractUserIdsFromResolution,
} from "@/lib/mention-resolution";
import { createMentionNotifications } from "@/lib/notifications";

const inputSchema = zfd.formData({
  title: zfd.text(z.string().min(1).max(255)),
  content: zfd.text(z.string().optional()),
  postType: zfd.text(
    z.enum(["user-post", "paper-post", "pure-paper"]).optional()
  ),
  paperId: zfd.text(z.string().optional()), // Added support for paperId
  images: zfd.text(z.string().optional()), // JSON string of image URLs
  mentions: zfd.text(z.string().optional()), // JSON string of mention strings
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

    // Parse and resolve mentions if provided
    let mentionedUserIds: string[] = [];
    let resolvedMentions: any[] = [];
    if (input.mentions) {
      try {
        const mentionStrings: string[] = JSON.parse(input.mentions);
        resolvedMentions = await resolveMentions(
          mentionStrings,
          session.user.id
        );
        mentionedUserIds = extractUserIdsFromResolution(
          resolvedMentions,
          session.user.id
        );

        console.log(
          `📧 Resolved ${mentionStrings.length} mentions to ${mentionedUserIds.length} users`
        );
      } catch (error) {
        console.error("Error parsing or resolving mentions:", error);
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

    // Create notifications for mentioned users
    if (resolvedMentions.length > 0) {
      try {
        const actorName = session.user.name || session.user.email?.split("@")[0] || "Someone";
        const actionUrl = `/posts/${post.id}`;
        
        await createMentionNotifications(
          resolvedMentions,
          session.user.id,
          actorName,
          "post",
          post.id,
          actionUrl
        );
      } catch (error) {
        console.error("Error creating mention notifications:", error);
        // Don't fail the post creation if notifications fail
      }
    }

    revalidatePath("/");

    return { id: post.id };
  });
