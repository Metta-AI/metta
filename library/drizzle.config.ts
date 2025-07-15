import dotenv from "dotenv";

import { defineConfig } from "drizzle-kit";

dotenv.config();
dotenv.config({ path: `.env.local`, override: true });

const url = process.env.DATABASE_URL;

if (!url) {
  throw new Error("DATABASE_URL is not set");
}

export default defineConfig({
  dialect: "postgresql",
  schema: "./src/lib/db/*",
  out: "./drizzle",
  dbCredentials: { url },
});
