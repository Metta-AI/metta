-- Remove legacy string array fields from Paper table in favor of relational Institution entities
-- The institutions field stored institution names as a string array
-- The institutionIds field was unused
-- Both are replaced by the paperInstitutions join table linking to Institution entities

-- Drop the legacy string array columns
ALTER TABLE "public"."paper" DROP COLUMN "institutions";
ALTER TABLE "public"."paper" DROP COLUMN "institutionIds";

