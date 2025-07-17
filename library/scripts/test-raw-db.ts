#!/usr/bin/env tsx

import * as dotenv from 'dotenv';
import { Pool } from 'pg';

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

async function testRawDatabase() {
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
  });

  try {
    console.log('🔍 Testing raw database connection...');
    console.log(`   DATABASE_URL: ${process.env.DATABASE_URL}`);
    
    const client = await pool.connect();
    
    // Test basic connection
    const result = await client.query('SELECT current_database(), current_schema()');
    console.log('✅ Raw database connection successful');
    console.log(`   Database: ${result.rows[0].current_database}`);
    console.log(`   Schema: ${result.rows[0].current_schema}`);
    
    // Test if paper table exists
    const tableResult = await client.query(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' AND table_name = 'paper'
    `);
    
    console.log(`   Paper table exists: ${tableResult.rows.length > 0}`);
    
    if (tableResult.rows.length > 0) {
      // Test selecting from paper table
      const paperResult = await client.query('SELECT COUNT(*) FROM paper');
      console.log(`   Papers in table: ${paperResult.rows[0].count}`);
    }
    
    client.release();
    
  } catch (error) {
    console.error('❌ Raw database test failed:', error);
    throw error;
  } finally {
    await pool.end();
  }
}

testRawDatabase().catch(console.error); 