-- AlterTable
ALTER TABLE "public"."post" ADD COLUMN     "images" TEXT[] DEFAULT ARRAY[]::TEXT[];
