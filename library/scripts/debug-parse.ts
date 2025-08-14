#!/usr/bin/env tsx

import * as fs from 'fs';
import * as path from 'path';

const usersDataPath = path.join(__dirname, '..', '..', '..', 'metta-library-mock', 'observatory', 'src', 'mockData', 'users.ts');

function debugUsersParse() {
  console.log('🔍 Debugging users file parsing...');
  console.log(`File path: ${usersDataPath}`);
  
  if (!fs.existsSync(usersDataPath)) {
    console.log('❌ File not found');
    return;
  }

  const fileContent = fs.readFileSync(usersDataPath, 'utf8');
  console.log('\n📄 First 200 characters of file:');
  console.log(fileContent.substring(0, 200));
  
  // Try the regex
  const usersMatch = fileContent.match(/export const mockUsers: User\[\] = (\[[\s\S]*?\]);/);
  
  if (!usersMatch) {
    console.log('\n❌ No match found');
    return;
  }
  
  console.log('\n✅ Match found!');
  console.log('📋 Extracted array (first 300 chars):');
  const extracted = usersMatch[1];
  console.log(extracted.substring(0, 300));
  
  console.log('\n🧹 After cleaning (first 300 chars):');
  const cleanedString = extracted
    .replace(/,(\s*[}\]])/g, '$1')
    .replace(/\n/g, ' ')
    .replace(/\s+/g, ' ');
  console.log(cleanedString.substring(0, 300));
  
  try {
    const parsed = JSON.parse(cleanedString);
    console.log('\n✅ Successfully parsed!');
    console.log(`📊 Found ${parsed.length} users`);
    console.log('👤 First user:', parsed[0]);
  } catch (error) {
    console.log('\n❌ JSON parse failed:', error);
  }
}

debugUsersParse(); 