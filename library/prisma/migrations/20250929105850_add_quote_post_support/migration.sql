-- AddQuotePostSupport
-- Add quotedPostIds field to support quote posts
ALTER TABLE "public"."post" ADD COLUMN "quotedPostIds" TEXT[] DEFAULT ARRAY[]::TEXT[];
