#!/usr/bin/env tsx

import * as dotenv from 'dotenv';
import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";

import * as authSchema from "../src/lib/db/schema/auth";
import * as postSchema from "../src/lib/db/schema/post";
import * as paperSchema from "../src/lib/db/schema/paper";

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

async function testMainConfig() {
  console.log('🔍 Testing with main db configuration...');
  
  // Create a connection pool
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
  });

  const db = drizzle(pool, {
    schema: { ...authSchema, ...postSchema, ...paperSchema },
  });
  
  try {
    // Test with papersTable from the schema
    const result = await db.select().from(paperSchema.papersTable).limit(1);
    console.log('✅ Main config query works:', result);
    
  } catch (error) {
    console.error('❌ Main config test failed:', error);
  } finally {
    await pool.end();
  }
}

testMainConfig().catch(console.error); 