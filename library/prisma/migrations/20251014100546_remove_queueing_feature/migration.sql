-- DropIndex
DROP INDEX IF EXISTS "user_paper_interaction_userId_queued_idx";

-- AlterTable
ALTER TABLE "user_paper_interaction" DROP COLUMN IF EXISTS "queued";

-- AlterTable
ALTER TABLE "post" DROP COLUMN IF EXISTS "queues";

