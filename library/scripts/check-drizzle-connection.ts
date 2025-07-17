#!/usr/bin/env tsx

import * as dotenv from 'dotenv';

// Load environment variables FIRST
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

// Then import db and schemas
import { db } from "../src/lib/db";

async function checkDrizzleConnection() {
  console.log('🔍 Checking Drizzle connection details...');
  console.log(`   DATABASE_URL: ${process.env.DATABASE_URL}`);
  
  try {
    // Use raw SQL to check what database we're connected to
    const result = await db.execute('SELECT current_database(), current_schema(), session_user');
    console.log('✅ Drizzle connection details:', result.rows[0]);
    
    // Check if paper table exists in this database
    const tableCheck = await db.execute(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      AND table_name = 'paper'
    `);
    console.log('📋 Paper table exists:', tableCheck.rows.length > 0);
    
  } catch (error) {
    console.error('❌ Connection check failed:', error);
  }
}

checkDrizzleConnection().catch(console.error); 