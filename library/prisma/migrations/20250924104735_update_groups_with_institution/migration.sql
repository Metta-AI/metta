-- DropIndex
DROP INDEX "group_name_key";

-- AlterTable
ALTER TABLE "group" ADD COLUMN "institutionId" TEXT NOT NULL DEFAULT '';

-- Update existing groups to use a default institution (if any exist)
-- Since we just reset the DB, this should be safe, but adding for completeness
UPDATE "group" SET "institutionId" = (
  SELECT id FROM "institution" LIMIT 1
) WHERE "institutionId" = '';

-- Remove the default now that we've updated existing rows
ALTER TABLE "group" ALTER COLUMN "institutionId" DROP DEFAULT;

-- CreateIndex
CREATE UNIQUE INDEX "group_name_institutionId_key" ON "group"("name", "institutionId");

-- AddForeignKey
ALTER TABLE "group" ADD CONSTRAINT "group_institutionId_fkey" FOREIGN KEY ("institutionId") REFERENCES "institution"("id") ON DELETE CASCADE ON UPDATE CASCADE;
