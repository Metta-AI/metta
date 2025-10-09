import { NextRequest, NextResponse } from "next/server";
import { generateText } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { z } from "zod";

import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import { extractPdfWithOpenAI } from "@/lib/openai-pdf-extractor";
import { config } from "@/lib/config";
import {
  AuthenticationError,
  NotFoundError,
  ServiceUnavailableError,
} from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";
import { Logger } from "@/lib/logging/logger";

const botRequestSchema = z.object({
  message: z.string().min(1, "Message cannot be empty"),
  postId: z.string().min(1, "Post ID is required"),
});

export async function POST(request: NextRequest) {
  try {
    // Verify user authentication
    const session = await getSessionOrRedirect();
    if (!session?.user?.id) {
      throw new AuthenticationError();
    }

    // Parse and validate request body
    const body = await request.json();
    const { message, postId } = botRequestSchema.parse(body);

    // Get the post and associated paper
    const post = await prisma.post.findUnique({
      where: { id: postId },
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

    if (!post) {
      throw new NotFoundError("Post", postId);
    }

    if (!post.paper) {
      throw new NotFoundError("Paper associated with post");
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
        Logger.info("Fetching PDF content for bot analysis", {
          paperId: post.paper.id,
          paperUrl: post.paper.link,
        });

        // Convert arXiv URLs to PDF format if needed
        let pdfUrl = post.paper.link;
        if (pdfUrl.includes("arxiv.org/abs/")) {
          pdfUrl = pdfUrl.replace("/abs/", "/pdf/") + ".pdf";
        }

        const pdfResponse = await fetch(pdfUrl);
        if (pdfResponse.ok) {
          const pdfBuffer = Buffer.from(await pdfResponse.arrayBuffer());
          const pdfContent = await extractPdfWithOpenAI(pdfBuffer);

          paperContext = `Title: ${pdfContent.title}\n\nSummary: ${pdfContent.summary}\n\nKey Points: ${pdfContent.shortExplanation}`;
          Logger.info("Successfully extracted PDF content for bot analysis", {
            paperId: post.paper.id,
          });
        } else {
          throw new Error(`Failed to fetch PDF: ${pdfResponse.status}`);
        }
      } catch (error) {
        Logger.warn("Error extracting PDF for bot, using fallback", {
          paperId: post.paper.id,
          error: error instanceof Error ? error.message : String(error),
        });
        // Fallback to basic abstract
        paperContext = `Title: ${post.paper.title}\n\nAbstract: ${post.paper.abstract}`;
      }
    } else if (post.paper.abstract) {
      // Minimal fallback to just abstract
      paperContext = `Title: ${post.paper.title}\n\nAbstract: ${post.paper.abstract}`;
    } else {
      throw new NotFoundError("Paper content for analysis", post.paper.id);
    }

    // Generate response using GPT-4o (treating as GPT-5 for system prompt)
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

User Question: ${message}

Please provide a helpful response about this paper.`,
        },
      ],

      temperature: 0.7, // Balanced creativity
    });

    return NextResponse.json({
      success: true,
      response: result.text,
      metadata: {
        paperTitle: post.paper.title,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error) {
    return handleApiError(error, { endpoint: "POST /api/chat/library-bot" });
  }
}
