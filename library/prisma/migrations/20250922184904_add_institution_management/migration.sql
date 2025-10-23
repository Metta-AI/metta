-- CreateEnum
CREATE TYPE "public"."InstitutionType" AS ENUM ('UNIVERSITY', 'COMPANY', 'RESEARCH_LAB', 'NONPROFIT', 'GOVERNMENT', 'OTHER');

-- CreateTable
CREATE TABLE "public"."institution" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "domain" TEXT,
    "description" TEXT,
    "website" TEXT,
    "location" TEXT,
    "type" "public"."InstitutionType" NOT NULL DEFAULT 'COMPANY',
    "isVerified" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdByUserId" TEXT,

    CONSTRAINT "institution_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."user_institution" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "institutionId" TEXT NOT NULL,
    "role" TEXT,
    "department" TEXT,
    "title" TEXT,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "joinedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "user_institution_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "institution_name_key" ON "public"."institution"("name");

-- CreateIndex
CREATE UNIQUE INDEX "institution_domain_key" ON "public"."institution"("domain");

-- CreateIndex
CREATE UNIQUE INDEX "user_institution_userId_institutionId_key" ON "public"."user_institution"("userId", "institutionId");

-- AddForeignKey
ALTER TABLE "public"."institution" ADD CONSTRAINT "institution_createdByUserId_fkey" FOREIGN KEY ("createdByUserId") REFERENCES "public"."user"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."user_institution" ADD CONSTRAINT "user_institution_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."user"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."user_institution" ADD CONSTRAINT "user_institution_institutionId_fkey" FOREIGN KEY ("institutionId") REFERENCES "public"."institution"("id") ON DELETE CASCADE ON UPDATE CASCADE;
