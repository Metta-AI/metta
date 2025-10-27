-- CreateIndex
CREATE INDEX "comment_postId_createdAt_idx" ON "public"."comment"("postId", "createdAt");

-- CreateIndex
CREATE INDEX "comment_authorId_idx" ON "public"."comment"("authorId");

-- CreateIndex
CREATE INDEX "notification_userId_isRead_createdAt_idx" ON "public"."notification"("userId", "isRead", "createdAt");

-- CreateIndex
CREATE INDEX "notification_type_createdAt_idx" ON "public"."notification"("type", "createdAt");

-- CreateIndex
CREATE INDEX "post_createdAt_id_idx" ON "public"."post"("createdAt", "id");

-- CreateIndex
CREATE INDEX "post_paperId_idx" ON "public"."post"("paperId");

-- CreateIndex
CREATE INDEX "post_authorId_createdAt_idx" ON "public"."post"("authorId", "createdAt");

-- CreateIndex
CREATE INDEX "user_paper_interaction_userId_starred_idx" ON "public"."user_paper_interaction"("userId", "starred");

-- CreateIndex
CREATE INDEX "user_paper_interaction_userId_queued_idx" ON "public"."user_paper_interaction"("userId", "queued");

-- CreateIndex
CREATE INDEX "user_paper_interaction_paperId_starred_idx" ON "public"."user_paper_interaction"("paperId", "starred");
