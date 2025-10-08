# API Routes Migration Plan

## Overview

Current state: 24 API route files
Goal: Remove redundant routes, keep only what's necessary

---

## Category 1: 🗑️ Redundant Read Routes (Can Remove)

These routes just fetch data and are redundant with server-side data functions already used in pages.

### 1. `/api/posts` ✂️

- **Usage**: Lists posts with pagination
- **Alternative**: `listPosts()` from `@/posts/data/posts-server` (already used in pages)
- **Action**: DELETE - already have server function

### 2. `/api/posts/[id]` ✂️

- **Usage**: Fetches single post
- **Alternative**: `loadPost()` from `@/posts/data/post` (used in pages)
- **Action**: DELETE - redundant

### 3. `/api/authors` ✂️

- **Usage**: Lists/searches authors
- **Alternative**: `loadAuthors()` from `@/posts/data/authors-server`
- **Action**: DELETE - redundant

### 4. `/api/authors/[authorId]` ✂️

- **Usage**: Fetches single author
- **Alternative**: `loadAuthor()` from `@/posts/data/authors-server`
- **Action**: DELETE - redundant

### 5. `/api/institutions` ✂️

- **Usage**: Lists institutions
- **Alternative**: Server components can call directly
- **Action**: DELETE - redundant

### 6. `/api/institutions/[name]` ✂️

- **Usage**: Fetches single institution
- **Alternative**: `loadInstitution()` from `@/posts/data/institutions-server`
- **Action**: DELETE - redundant

### 7. `/api/papers/[postId]/data` ✂️

- **Usage**: Fetches paper data
- **Alternative**: Direct Prisma query or dedicated function
- **Action**: DELETE or migrate to server function

### 8. `/api/papers/[postId]/institutions` ✂️

- **Usage**: Fetches paper institutions
- **Alternative**: Include in main paper query
- **Action**: DELETE - redundant

---

## Category 2: 🔄 Mutation Routes (Migrate to Server Actions)

These routes perform mutations but should use Server Actions instead.

### 9. `/api/notifications/mark-read` → Server Action ✅

- **Current**: POST endpoint
- **Alternative**: Create `markNotificationsReadAction()`
- **Action**: MIGRATE to server action, then DELETE route

### 10. `/api/notification-preferences` (GET/PUT) → Mixed

- **GET**: Can be server function
- **PUT**: Convert to server action `updateNotificationPreferencesAction()`
- **Action**: MIGRATE, then DELETE route

---

## Category 3: ✅ Keep (Special Purpose)

These routes have legitimate reasons to remain as API routes.

### File Handling (Keep)

- `/api/upload-image` ✅ - File upload, needs multipart/form-data
- `/api/analyze-pdf` ✅ - PDF processing with streaming
- `/api/download-pdf` ✅ - File download/proxy

### OAuth & External Integrations (Keep)

- `/api/auth/[...nextauth]` ✅ - NextAuth required route
- `/api/discord/auth` ✅ - OAuth callback
- `/api/discord/link` ✅ - OAuth callback
- `/api/discord/test` ✅ - Testing/debugging

### Streaming/Real-time (Keep)

- `/api/chat/library-bot` ✅ - Streaming AI responses

### Testing/Debug (Keep for now)

- `/api/email-test` ⚠️ - Testing (could remove in prod)

### Mentions (Keep)

- `/api/mentions/search` ✅ - Used by mention picker UI

---

## Category 4: 🤔 Admin Routes (Evaluate)

Could migrate but lower priority.

- `/api/admin/users` ⚠️ - Admin panel, could stay or migrate
- `/api/admin/institutions` ⚠️ - Admin panel, could stay or migrate

---

## Category 5: 📊 Notifications (Complex)

### Current State

- `/api/notifications` (GET) - Fetch notifications
- `/api/notifications/count` (GET) - Fetch count
- `/api/notifications/mark-read` (POST) - Mark as read

### Recommendation

- **Keep `/api/notifications` and `/api/notifications/count`** - Used by client polling
- **Migrate `/api/notifications/mark-read`** → Server Action

---

## Migration Summary

### Phase 1: Remove Redundant Read Routes (Quick Win - 1 hour)

Delete these 8 routes (already have alternatives):

1. ✂️ `/api/posts`
2. ✂️ `/api/posts/[id]`
3. ✂️ `/api/authors`
4. ✂️ `/api/authors/[authorId]`
5. ✂️ `/api/institutions`
6. ✂️ `/api/institutions/[name]`
7. ✂️ `/api/papers/[postId]/data`
8. ✂️ `/api/papers/[postId]/institutions`

**Impact**: -200 lines of code, cleaner API surface

### Phase 2: Migrate Mutations to Server Actions (2 hours)

1. Create `markNotificationsReadAction()`
2. Create `updateNotificationPreferencesAction()` (already exists?)
3. Delete corresponding API routes

**Impact**: More consistent mutation patterns

### Phase 3: Verify No Usage (30 min)

- Search codebase for API route usage
- Check `fetch()` calls in client components
- Update any found usages

---

## Verification Steps

Before deleting routes:

```bash
# Search for API route usage in client code
rg "'/api/posts'" --type ts --type tsx
rg "'/api/authors'" --type ts --type tsx
rg "'/api/institutions'" --type ts --type tsx
rg "'/api/papers/.*data'" --type ts --type tsx
rg "'/api/notifications/mark-read'" --type ts --type tsx
```

---

## Final Count

**Current**: 24 API routes
**After cleanup**: ~14 API routes (42% reduction)

**Delete**: 8-10 routes
**Keep**: 12-14 routes
**Migrate then delete**: 2 routes
