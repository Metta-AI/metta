-- AlterTable
ALTER TABLE "public"."author" ADD COLUMN     "institutionId" TEXT;

-- AlterTable
ALTER TABLE "public"."paper" ADD COLUMN     "institutionIds" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- CreateTable
CREATE TABLE "public"."paper_institution" (
    "id" TEXT NOT NULL,
    "paperId" TEXT NOT NULL,
    "institutionId" TEXT NOT NULL,

    CONSTRAINT "paper_institution_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "paper_institution_paperId_institutionId_key" ON "public"."paper_institution"("paperId", "institutionId");

-- AddForeignKey
ALTER TABLE "public"."author" ADD CONSTRAINT "author_institutionId_fkey" FOREIGN KEY ("institutionId") REFERENCES "public"."institution"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."paper_institution" ADD CONSTRAINT "paper_institution_paperId_fkey" FOREIGN KEY ("paperId") REFERENCES "public"."paper"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."paper_institution" ADD CONSTRAINT "paper_institution_institutionId_fkey" FOREIGN KEY ("institutionId") REFERENCES "public"."institution"("id") ON DELETE CASCADE ON UPDATE CASCADE;
