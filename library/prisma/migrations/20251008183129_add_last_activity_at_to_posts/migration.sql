-- AlterTable
ALTER TABLE "public"."post" ADD COLUMN     "lastActivityAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP;

-- CreateIndex
CREATE INDEX "post_lastActivityAt_id_idx" ON "public"."post"("lastActivityAt", "id");
