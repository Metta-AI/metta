"use server";

import { revalidatePath } from "next/cache";
import { zfd } from "zod-form-data";
import { z } from "zod/v4";

import { actionClient } from "@/lib/actionClient";
import { prisma } from "@/lib/db/prisma";
import { config } from "@/lib/config";
import { generateText } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { extractPdfContent } from "@/lib/pdf-extractor";

const inputSchema = zfd.formData({
  postId: zfd.text(z.string().min(1)),
  parentCommentId: zfd.text(z.string().min(1)),
  userMessage: zfd.text(z.string().min(1)),
});

export const createBotResponseAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    try {
      // Get the paper data for context
      const post = await prisma.post.findUnique({
        where: { id: input.postId },
        include: {
          paper: {
            select: {
              id: true,
              title: true,
              abstract: true,
              link: true,
              llmAbstract: true,
            },
          },
        },
      });

      if (!post?.paper) {
        throw new Error("No paper found for this post");
      }

      // Prepare paper context for the LLM
      let paperContext = "";

      if (post.paper.llmAbstract) {
        // Use LLM-generated abstract if available (most comprehensive)
        const llmContent = post.paper.llmAbstract as any;
        paperContext = `Title: ${post.paper.title}\n\nAbstract: ${post.paper.abstract}\n\nDetailed Analysis: ${llmContent.summary || llmContent.explanation || JSON.stringify(llmContent)}`;
      } else if (post.paper.link && config.llm.anthropicApiKey) {
        // Try to extract PDF content on-demand
        try {
          console.log(
            `📄 Fetching PDF content for bot analysis: ${post.paper.link}`
          );

          // Convert arXiv URLs to PDF format if needed
          let pdfUrl = post.paper.link;
          if (pdfUrl.includes("arxiv.org/abs/")) {
            pdfUrl = pdfUrl.replace("/abs/", "/pdf/") + ".pdf";
          }

          const pdfResponse = await fetch(pdfUrl);
          if (pdfResponse.ok) {
            const pdfBuffer = Buffer.from(await pdfResponse.arrayBuffer());
            const pdfContent = await extractPdfContent(pdfBuffer);

            paperContext = `Title: ${pdfContent.title}\n\nSummary: ${pdfContent.summary}\n\nKey Points: ${pdfContent.shortExplanation}`;
            console.log(
              `✅ Successfully extracted PDF content for bot analysis`
            );
          } else {
            throw new Error(`Failed to fetch PDF: ${pdfResponse.status}`);
          }
        } catch (pdfError) {
          console.error(`❌ Error extracting PDF for bot:`, pdfError);
          // Fallback to basic abstract
          paperContext = `Title: ${post.paper.title}\n\nAbstract: ${post.paper.abstract}`;
        }
      } else if (post.paper.abstract) {
        // Minimal fallback to just abstract
        paperContext = `Title: ${post.paper.title}\n\nAbstract: ${post.paper.abstract}`;
      } else {
        throw new Error("No paper content available for analysis");
      }

      // Generate response using GPT-4o
      const systemPrompt = `You are @library_bot, an AI assistant specialized in discussing academic papers. You are responding to a user's question about a specific research paper in a discussion forum format.

Your response should be:
- Accurate and based on the paper content provided
- Conversational and engaging, as if participating in an academic discussion
- Focused on the user's specific question
- Well-formatted for a chat/comment interface
- Professional but approachable in tone

Remember: You are responding as @library_bot in a discussion thread, so your response should feel natural in that context.

The paper content is provided below for reference.`;

      const result = await generateText({
        model: anthropic("claude-3-5-sonnet-20241022"), // Using Claude 3.5 Sonnet as the most advanced available
        messages: [
          {
            role: "system",
            content: systemPrompt,
          },
          {
            role: "user",
            content: `Paper Content:
${paperContext}

---

User Question: ${input.userMessage}

Please provide a helpful response about this paper.`,
          },
        ],

        temperature: 0.7, // Balanced creativity
      });

      // Create a system user for the bot if it doesn't exist
      let botUser = await prisma.user.findFirst({
        where: { email: "library_bot@system" },
      });

      if (!botUser) {
        botUser = await prisma.user.create({
          data: {
            name: "Library Bot",
            email: "library_bot@system",
            image: null,
          },
        });
      }

      // Create the bot response comment
      const botComment = await prisma.comment.create({
        data: {
          content: result.text,
          postId: input.postId,
          parentId: input.parentCommentId, // Reply to the user's comment
          authorId: botUser.id,
          isBot: true,
          botType: "library_bot",
        },
        include: {
          author: true,
        },
      });

      // Update the post's reply count
      await prisma.post.update({
        where: { id: input.postId },
        data: {
          replies: {
            increment: 1,
          },
        },
      });

      // Revalidate the current page to show updated data
      revalidatePath("/");

      return {
        success: true,
        botComment: {
          id: botComment.id,
          content: botComment.content,
          postId: botComment.postId,
          parentId: botComment.parentId,
          isBot: true,
          botType: "library_bot",
          author: {
            id: botComment.author.id,
            name: botComment.author.name,
            email: botComment.author.email,
            image: botComment.author.image,
          },
          createdAt: botComment.createdAt,
          updatedAt: botComment.updatedAt,
        },
      };
    } catch (error) {
      console.error("Error generating bot response:", error);

      // Create a fallback bot response
      let botUser = await prisma.user.findFirst({
        where: { email: "library_bot@system" },
      });

      if (!botUser) {
        botUser = await prisma.user.create({
          data: {
            name: "Library Bot",
            email: "library_bot@system",
            image: null,
          },
        });
      }

      const fallbackComment = await prisma.comment.create({
        data: {
          content:
            "I'm having trouble processing your request right now. Please try again later.",
          postId: input.postId,
          parentId: input.parentCommentId,
          authorId: botUser.id,
          isBot: true,
          botType: "library_bot",
        },
        include: {
          author: true,
        },
      });

      // Update the post's reply count
      await prisma.post.update({
        where: { id: input.postId },
        data: {
          replies: {
            increment: 1,
          },
        },
      });

      revalidatePath("/");

      return {
        success: true,
        botComment: {
          id: fallbackComment.id,
          content: fallbackComment.content,
          postId: fallbackComment.postId,
          parentId: fallbackComment.parentId,
          isBot: true,
          botType: "library_bot",
          author: {
            id: fallbackComment.author.id,
            name: fallbackComment.author.name,
            email: fallbackComment.author.email,
            image: fallbackComment.author.image,
          },
          createdAt: fallbackComment.createdAt,
          updatedAt: fallbackComment.updatedAt,
        },
      };
    }
  });
