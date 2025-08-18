import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local", quiet: true }); // ← load variables first

import { PrismaClient } from "@prisma/client";

// Create a singleton Prisma client instance
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

export const prisma = globalForPrisma.prisma ?? new PrismaClient();

if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}
