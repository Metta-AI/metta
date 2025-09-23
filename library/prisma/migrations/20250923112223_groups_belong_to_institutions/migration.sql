-- CreateForeignKey
-- This migration makes groups belong to institutions

-- Step 1: Add institutionId column as nullable
ALTER TABLE "group" ADD COLUMN "institutionId" TEXT;

-- Step 2: Populate institutionId for existing groups
-- Assign existing groups to Softmax institution (the user-created one)
UPDATE "group" 
SET "institutionId" = (
  SELECT id FROM "institution" WHERE name = 'Softmax' LIMIT 1
)
WHERE "institutionId" IS NULL;

-- Step 3: Make institutionId required
ALTER TABLE "group" ALTER COLUMN "institutionId" SET NOT NULL;

-- Step 4: Add foreign key constraint
ALTER TABLE "group" ADD CONSTRAINT "group_institutionId_fkey" FOREIGN KEY ("institutionId") REFERENCES "institution"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Step 5: Add compound unique constraint (name must be unique within institution)
ALTER TABLE "group" ADD CONSTRAINT "group_name_institutionId_key" UNIQUE ("name", "institutionId");
