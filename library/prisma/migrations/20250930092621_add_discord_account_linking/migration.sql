-- AlterTable
ALTER TABLE "public"."notification_preference" ADD COLUMN     "discordLinkedAt" TIMESTAMP(3),
ADD COLUMN     "discordUserId" TEXT,
ADD COLUMN     "discordUsername" TEXT;

-- CreateIndex
CREATE INDEX "notification_preference_discordUserId_idx" ON "public"."notification_preference"("discordUserId");

-- CreateIndex
CREATE UNIQUE INDEX "notification_preference_discordUserId_type_key" ON "public"."notification_preference"("discordUserId", "type");
