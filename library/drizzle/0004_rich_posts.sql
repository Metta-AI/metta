-- Migration: Add rich post features for social feed
-- This migration extends the post table to support rich social feed functionality
-- including content with LaTeX support, post types, social metrics, and paper references

-- Add new columns to the post table
ALTER TABLE "post" ADD COLUMN "content" text;
ALTER TABLE "post" ADD COLUMN "postType" text DEFAULT 'user-post' CHECK ("postType" IN ('user-post', 'paper-post', 'pure-paper'));
ALTER TABLE "post" ADD COLUMN "likes" integer DEFAULT 0;
ALTER TABLE "post" ADD COLUMN "retweets" integer DEFAULT 0;
ALTER TABLE "post" ADD COLUMN "replies" integer DEFAULT 0;
ALTER TABLE "post" ADD COLUMN "paperId" text REFERENCES "paper"("id") ON DELETE SET NULL;

-- Create index for post type filtering
CREATE INDEX "post_type_idx" ON "post"("postType");

-- Create index for paper reference lookups
CREATE INDEX "post_paper_id_idx" ON "post"("paperId");

-- Create index for social metrics sorting
CREATE INDEX "post_social_metrics_idx" ON "post"("likes", "retweets", "replies");

-- Add comment explaining the new schema
COMMENT ON COLUMN "post"."content" IS 'The main content of the post, supports LaTeX with $...$ and $$...$$ syntax';
COMMENT ON COLUMN "post"."postType" IS 'Type of post: user-post (text only), paper-post (text + paper), pure-paper (paper only)';
COMMENT ON COLUMN "post"."likes" IS 'Number of likes on this post';
COMMENT ON COLUMN "post"."retweets" IS 'Number of retweets/shares of this post';
COMMENT ON COLUMN "post"."replies" IS 'Number of replies to this post';
COMMENT ON COLUMN "post"."paperId" IS 'Reference to a paper if this post is about a specific paper'; 