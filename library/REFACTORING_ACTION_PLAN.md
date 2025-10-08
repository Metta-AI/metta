# Backend Refactoring - Quick Action Plan

**Priority Order: Fix → Improve → Optimize → Polish**

---

## 🔥 Critical Issues (Do First)

### 1. Fix Prisma Client Duplication

**Time:** 1 hour
**Impact:** High
**Difficulty:** Low

```bash
# Steps:
1. Delete src/lib/db/index.ts
2. Keep src/lib/db/prisma.ts as the single source
3. Search and replace any imports from "@/lib/db" to "@/lib/db/prisma"
```

**Files to change:**

- `src/app/api/papers/[postId]/data/route.ts` (uses `db`)
- Any other files importing from `@/lib/db`

---

### 2. Add Database Indices

**Time:** 30 minutes
**Impact:** High (Performance)
**Difficulty:** Low

```prisma
// Add to prisma/schema.prisma

model Post {
  // ... existing fields

  @@index([createdAt, id])
  @@index([paperId])
  @@index([authorId, createdAt])
}

model Notification {
  // ... existing fields

  @@index([userId, isRead, createdAt])
  @@index([type, createdAt])
}

model UserPaperInteraction {
  // ... existing fields

  @@index([userId, starred])
  @@index([userId, queued])
  @@index([paperId, starred])
}

model Comment {
  // ... existing fields

  @@index([postId, createdAt])
  @@index([authorId])
}
```

```bash
# Run migration
pnpm prisma migrate dev --name add_performance_indices
```

---

## 📊 High Priority Improvements

### 3. Create Base Worker Class

**Time:** 2-3 hours
**Impact:** Medium (Maintainability)
**Difficulty:** Medium

**Files to create:**

- `src/lib/workers/base-worker.ts`

**Files to refactor:**

- `src/lib/workers/institution-worker.ts`
- `src/lib/workers/llm-worker.ts`
- `src/lib/workers/tagging-worker.ts`
- `src/lib/workers/external-notification-worker.ts`

See detailed implementation in main analysis document, section 4.

---

### 4. Standardize Error Handling

**Time:** 2-3 hours
**Impact:** Medium (DX & Debugging)
**Difficulty:** Medium

**Files to create:**

```typescript
// src/lib/errors.ts
export class AppError extends Error {
  /* ... */
}
export class ValidationError extends AppError {
  /* ... */
}
export class NotFoundError extends AppError {
  /* ... */
}
export class AuthorizationError extends AppError {
  /* ... */
}

// src/lib/api/error-handler.ts
export function handleApiError(error: unknown): NextResponse {
  /* ... */
}
```

**Then update:**

- All API routes to use `handleApiError`
- All server actions to use custom error classes

---

### 5. Create Configuration Service

**Time:** 1-2 hours
**Impact:** Medium (Maintainability)
**Difficulty:** Low

**File to create:**

- `src/lib/config.ts`

**Then update:**

- `src/lib/job-queue.ts` - use `config.redis`
- `src/lib/auth.ts` - use `config.auth`
- All files accessing `process.env` directly

---

## 🎯 Medium Priority

### 6. Refactor Institution Actions to Use Service Layer

**Time:** 4-6 hours
**Impact:** Medium (Maintainability)
**Difficulty:** Medium-High

**Example: Institution Membership**

```
Create structure:
src/institutions/
  ├── actions/          (keep existing)
  ├── services/         (NEW)
  │   └── membership-service.ts
  └── data/             (NEW)
      └── institution-repository.ts
```

**Pattern to follow:**

1. Create repository for data access
2. Create service for business logic
3. Slim down actions to orchestration only

**Start with:** `joinInstitutionAction.ts`

---

### 7. Simplify Notification System

**Time:** 6-8 hours
**Impact:** Medium (Maintainability & Performance)
**Difficulty:** High

**Files to refactor:**

- `src/lib/notifications.ts`
- `src/lib/external-notifications/email.ts`
- `src/lib/external-notifications/discord-bot.ts`
- `src/lib/workers/external-notification-worker.ts`

**Goals:**

- Single point of entry for notifications
- Reduce database queries
- Simplify channel abstraction

---

### 8. Remove Redundant API Routes

**Time:** 2-3 hours
**Impact:** Low-Medium (Code Clarity)
**Difficulty:** Low

**Routes to evaluate for removal:**

```bash
# Can likely be removed (use Server Actions instead):
src/app/api/posts/route.ts
src/app/api/notifications/route.ts
src/app/api/mentions/search/route.ts

# Should keep (special functionality):
src/app/api/auth/[...nextauth]/route.ts  ✓
src/app/api/upload-image/route.ts        ✓
src/app/api/download-pdf/route.ts        ✓
src/app/api/discord/*                    ✓
src/app/api/admin/*                      ✓
```

**Before removing, verify:**

1. No external consumers
2. Functionality available via Server Action
3. Update any API client code

---

## 🚀 Optimization (Do After Above)

### 9. Optimize Feed Query

**Time:** 2-3 hours
**Impact:** High (Performance)
**Difficulty:** Medium

**File to optimize:**

- `src/posts/data/feed.ts`

**Issues:**

- Loads all posts, sorts in memory
- Separate query for quoted posts
- Heavy nested includes

**Solution:**

```typescript
// Use proper cursor pagination
// Limit includes to what's needed
// Consider caching for hot feed items
```

---

### 10. Optimize Institution Aggregations

**Time:** 2-3 hours
**Impact:** Medium (Performance)
**Difficulty:** Medium

**File to optimize:**

- `src/posts/data/institutions-server.ts`

**Issues:**

- Loads all data, aggregates in memory
- Heavy nested includes

**Solution:**

```typescript
// Use Prisma aggregation functions
// Consider raw SQL for complex aggregations
// Add caching layer for stats
```

---

## 📝 Polish (Lower Priority)

### 11. Set Up Testing Infrastructure

**Time:** 4-6 hours
**Impact:** Medium (Long-term)
**Difficulty:** Low-Medium

```bash
pnpm add -D vitest @testing-library/react @testing-library/jest-dom
```

**Files to create:**

- `vitest.config.ts`
- `vitest.setup.ts`
- Start with: `src/lib/notifications/__tests__/`

---

### 12. Improve Type Safety

**Time:** Ongoing
**Impact:** Medium (DX)
**Difficulty:** Low

**Actions:**

- Replace `any` types
- Add return type annotations
- Use Prisma select for type inference
- Create shared type definitions

---

## 📈 Metrics to Track

### Before/After Metrics

| Metric                       | Before | Target |
| ---------------------------- | ------ | ------ |
| Unique Prisma imports        | 2      | 1      |
| API routes                   | 24     | ~15    |
| Avg DB queries per feed load | ~8     | ~4     |
| Feed load time (p95)         | ?      | <500ms |
| Test coverage                | 0%     | >70%   |
| TypeScript errors            | ?      | 0      |

---

## Weekly Sprint Plan

### Week 1: Foundation

- [ ] Fix Prisma duplication (#1)
- [ ] Add database indices (#2)
- [ ] Create error handling (#4)
- [ ] Create config service (#5)

### Week 2: Services

- [ ] Create base worker class (#3)
- [ ] Refactor institution actions (#6)
- [ ] Start testing infrastructure (#11)

### Week 3: Cleanup

- [ ] Simplify notifications (#7)
- [ ] Remove redundant routes (#8)
- [ ] Add tests for new services

### Week 4: Optimization

- [ ] Optimize feed query (#9)
- [ ] Optimize institution queries (#10)
- [ ] Performance profiling

### Week 5+: Polish & Monitor

- [ ] Improve type safety (#12)
- [ ] Add integration tests
- [ ] Monitor metrics
- [ ] Iterate based on findings

---

## Quick Wins (Do Today!)

### 30-Minute Wins

1. ✅ Add database indices (#2)
2. ✅ Fix Prisma import duplication (#1)
3. ✅ Remove unused imports

### 1-Hour Wins

1. ✅ Create config service (#5)
2. ✅ Add ESLint rule for consistent imports
3. ✅ Document current API routes

---

## Commands Cheat Sheet

```bash
# Database
pnpm prisma migrate dev --name your_migration_name
pnpm prisma generate
pnpm prisma studio

# Development
pnpm dev              # Start Next.js
pnpm workers          # Start background workers
pnpm worker:dev       # Watch mode for workers

# Code Quality
pnpm lint
pnpm format
pnpm typecheck

# Testing (after setup)
pnpm test
pnpm test:watch
pnpm test:coverage
```

---

## Getting Help

If you get stuck:

1. Check main analysis: `BACKEND_REFACTORING_ANALYSIS.md`
2. Review Prisma docs: https://www.prisma.io/docs
3. Check Next.js docs: https://nextjs.org/docs

---

**Last Updated:** October 8, 2025
