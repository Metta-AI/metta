-- CreateEnum
CREATE TYPE "UserInstitutionStatus" AS ENUM ('PENDING', 'APPROVED', 'REJECTED');

-- AlterTable
ALTER TABLE "institution" ADD COLUMN "requiresApproval" BOOLEAN NOT NULL DEFAULT false;

-- AlterTable
ALTER TABLE "user_institution" ADD COLUMN "status" "UserInstitutionStatus" NOT NULL DEFAULT 'APPROVED';
