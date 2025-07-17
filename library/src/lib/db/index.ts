import * as dotenv from "dotenv";
dotenv.config({ path: ".env.local" });      // ← load variables first

import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";

import * as authSchema from "./schema/auth";
import * as postSchema from "./schema/post";
import * as paperSchema from "./schema/paper";

// Create a connection pool
const pool = new Pool({ connectionString: process.env.DATABASE_URL! });
export const db = drizzle(pool, {
  schema: { ...authSchema, ...postSchema, ...paperSchema },
});
