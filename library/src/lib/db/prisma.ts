import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local", quiet: true }); // ← load variables first

import { PrismaClient } from "@prisma/client";
import { config } from "../config";

// Create a singleton Prisma client instance
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

export const prisma = globalForPrisma.prisma ?? new PrismaClient();

if (config.nodeEnv !== "production") {
  globalForPrisma.prisma = prisma;
}
