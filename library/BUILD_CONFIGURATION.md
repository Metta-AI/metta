# Build Configuration Documentation

## Overview
This document captures the working configuration for the LibraryApp build system. These settings have been tested and verified to work correctly.

## Key Configuration Files

### PostCSS Configuration (`postcss.config.mjs`)
**Working Configuration:**
```javascript
const config = {
  plugins: ["@tailwindcss/postcss"],
};

export default config;
```

**Important Notes:**
- Uses **array syntax** `["@tailwindcss/postcss"]` - NOT object syntax
- This is the correct format for Tailwind CSS v4
- Do not change to `{"@tailwindcss/postcss": {}}` - this breaks CSS compilation

### Tailwind CSS Configuration
- **Version:** v4 (`"tailwindcss": "^4"`)
- **PostCSS Plugin:** `"@tailwindcss/postcss": "^4"`
- **Import Syntax:** `@import "tailwindcss";` in `src/app/globals.css`
- **No separate config file needed** - Tailwind v4 uses the PostCSS plugin

### CSS Structure (`src/app/globals.css`)
```css
@import "tailwindcss";

body {
  font-family: Arial, Helvetica, sans-serif;
}

/* Primary color classes for the sidebar */
.bg-primary-50 { background-color: #eff6ff; }
.bg-primary-100 { background-color: #dbeafe; }
.bg-primary-500 { background-color: #3b82f6; }
.bg-primary-600 { background-color: #2563eb; }
.text-primary-500 { color: #3b82f6; }
.text-primary-600 { color: #2563eb; }
.text-primary-700 { color: #1d4ed8; }
.border-primary-200 { border-color: #bfdbfe; }
.border-primary-500 { border-color: #3b82f6; }
```

## Build Process

### Prerequisites
1. **PostgreSQL:** Running via Postgres.app
2. **Node.js:** Version 18 or higher
3. **Package Manager:** pnpm

### Development Setup
```bash
# Install dependencies
pnpm install

# Generate Prisma client
pnpm prisma generate

# Start development server (run in separate terminal)
pnpm dev
```

### Build Commands
```bash
# Production build
pnpm build

# Clear build cache (if issues occur)
rm -rf .next
pnpm build
```

## Common Issues and Solutions

### CSS Loading Errors (404 for layout.css)
**Symptoms:** Browser console shows 404 errors for `/_next/static/css/app/layout.css`

**Root Cause:** Incorrect PostCSS configuration syntax
- ❌ Wrong: `plugins: {"@tailwindcss/postcss": {}}`
- ✅ Correct: `plugins: ["@tailwindcss/postcss"]`

**Solution:** Revert to array syntax in `postcss.config.mjs`

### Prisma Client Errors on Client Side
**Symptoms:** "Cannot find module" errors related to Prisma in browser

**Root Cause:** Server-side Prisma code being imported in client components

**Solution:** Use separate files for server and client code:
- Server: `src/posts/data/authors-server.ts`
- Client: `src/posts/data/authors-client.ts`

### Build Cache Issues
**Symptoms:** Stale builds, missing routes, or unexpected behavior

**Solution:** Clear build cache
```bash
rm -rf .next
pnpm build
```

## Architecture Patterns

### Server/Client Code Separation
- **Server files:** Use Prisma, run on server only
- **Client files:** Use fetch API, run in browser
- **Shared types:** Define in client file, import in server file

### File Naming Convention
- `*-server.ts` - Server-side only code
- `*-client.ts` - Client-side only code
- `route.ts` - API endpoints

## Verification Checklist

Before committing configuration changes:
- [ ] `pnpm build` completes successfully
- [ ] No TypeScript errors
- [ ] CSS loads without 404 errors
- [ ] Development server starts without errors
- [ ] All routes are included in build output

## Dependencies

### Core Dependencies
```json
{
  "next": "15.3.1",
  "react": "^19.0.0",
  "react-dom": "^19.0.0",
  "tailwindcss": "^4",
  "@tailwindcss/postcss": "^4",
  "@prisma/client": "^6.12.0"
}
```

### Development Dependencies
```json
{
  "prisma": "^6.12.0",
  "typescript": "^5"
}
```

## Environment Variables

### Required (.env.local)
```
DATABASE_URL=postgres://localhost/metta_library
DEV_MODE=true
AUTH_SECRET=<generated-secret>
```

## Notes for Future Development

1. **Always test builds** after configuration changes
2. **Use git to track** configuration changes
3. **Document breaking changes** in this file
4. **Separate server/client code** to avoid Prisma client errors
5. **Clear build cache** when troubleshooting

---

*Last Updated: [Current Date]*
*Working Build: ✓ Verified* 