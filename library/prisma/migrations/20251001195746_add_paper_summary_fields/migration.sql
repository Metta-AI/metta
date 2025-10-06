-- AlterTable
ALTER TABLE "public"."paper" ADD COLUMN     "abstractSummary" TEXT,
ADD COLUMN     "arxivUrl" TEXT,
ADD COLUMN     "citationCount" INTEGER NOT NULL DEFAULT 0,
ADD COLUMN     "doi" TEXT,
ADD COLUMN     "viewCount" INTEGER NOT NULL DEFAULT 0,
ALTER COLUMN "institutions" SET DEFAULT ARRAY[]::TEXT[],
ALTER COLUMN "tags" SET DEFAULT ARRAY[]::TEXT[];

-- CreateIndex
CREATE INDEX "paper_citationCount_idx" ON "public"."paper"("citationCount");
