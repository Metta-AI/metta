# Error Handling Migration Status

**Created:** October 8, 2025
**Last Updated:** October 8, 2025 (Latest)

## ✅ Completed

### Core Infrastructure (100%)

- ✅ Error class hierarchy (`src/lib/errors.ts`) - 10 typed error classes
- ✅ API error handler (`src/lib/api/error-handler.ts`) - handleApiError & withErrorHandler
- ✅ Logger service (`src/lib/logging/logger.ts`) - Structured logging with dev/prod modes
- ✅ Updated `actionClient.ts` to use new errors
- ✅ Created centralized API exports (`src/lib/api/index.ts`)

### API Routes Migrated (13/22 = 59%)

- ✅ `/api/notifications`
- ✅ `/api/notifications/count`
- ✅ `/api/notifications/mark-read`
- ✅ `/api/notification-preferences`
- ✅ `/api/posts`
- ✅ `/api/posts/[id]`
- ✅ `/api/papers/[postId]/data`
- ✅ `/api/papers/[postId]/institutions`
- ✅ `/api/institutions`
- ✅ `/api/institutions/[name]`
- ✅ `/api/authors`
- ✅ `/api/authors/[authorId]`
- ✅ `/api/mentions/search`

## 🚧 Remaining Work

### API Routes to Migrate (9 remaining)

1. `/api/admin/institutions/route.ts`
2. `/api/admin/users/route.ts`
3. `/api/discord/auth/route.ts`
4. `/api/discord/link/route.ts`
5. `/api/discord/test/route.ts`
6. `/api/email-test/route.ts`
7. `/api/upload-image/route.ts`
8. `/api/analyze-pdf/route.ts`
9. `/api/chat/library-bot/route.ts`

### console.error Replacement (201 instances across 66 files)

**High Priority Backend Files:**

- `lib/workers/*.ts` (5 files) - Background job workers
- `lib/external-notifications/*.ts` (2 files) - Email & Discord services
- `lib/arxiv-auto-import.ts` - Paper import service
- `lib/notifications.ts` - Notification creation
- `lib/mention-resolution.ts` - Mention processing
- `lib/paper-abstract-service.ts` - LLM abstract generation
- `lib/auto-tagging-service.ts` - Auto-tagging service
- `posts/actions/*.ts` (3 files) - Post/comment actions

**Lower Priority (Client-Side Components):**

- 40+ React component files with console.error
- Most are for client-side error display, less critical

### Server Actions (Optional)

- Consider using typed errors in server actions where appropriate
- Already using actionClient which handles errors, but could be more specific

## 📋 Migration Patterns

### Pattern 1: withErrorHandler (Simple Routes)

```typescript
import { withErrorHandler } from "@/lib/api/error-handler";

export const GET = withErrorHandler(async (request) => {
  const data = await fetchData();
  return NextResponse.json(data);
});
```

### Pattern 2: handleApiError (Complex Routes)

```typescript
import { handleApiError, AuthenticationError } from "@/lib/errors";

export async function GET(request) {
  try {
    if (!authenticated) throw new AuthenticationError();
    const data = await fetchData();
    return NextResponse.json(data);
  } catch (error) {
    return handleApiError(error, { endpoint: "GET /api/..." });
  }
}
```

### Pattern 3: Logger (Backend Services)

```typescript
import { Logger } from "@/lib/logging/logger";

// Replace:
console.error("Error message:", error);

// With:
Logger.error("Error message", error, { userId, action: "createPost" });
```

## 📊 Progress Summary

| Component           | Progress    | Status         |
| ------------------- | ----------- | -------------- |
| **Infrastructure**  | 100% (5/5)  | ✅ Complete    |
| **API Routes**      | 59% (13/22) | 🚧 In Progress |
| **Backend Logging** | 0% (0/201)  | ⏳ Not Started |
| **Server Actions**  | 0%          | ⏳ Not Started |

**Overall Progress: ~30% complete**

## 🎯 Next Steps

### Option A: Complete API Routes First

- Finish remaining 9 API routes (~1-2 hours)
- All API error handling will be consistent
- Good stopping point for commit/deploy

### Option B: Focus on Backend Logging

- Replace console.error with Logger in priority files
- Better observability and debugging
- Can be done incrementally

### Option C: Combination Approach

- Finish API routes (high traffic, user-facing)
- Then tackle backend logging (internal, debugging)
- Leave client components for later

## 💡 Recommendations

1. **Finish API routes next** - Only 9 files remaining, high user impact
2. **Then backend logging** - Systematic replacement in lib/ and workers/
3. **Test error scenarios** - Verify new error responses work correctly
4. **Set up error tracking** - Integrate with Sentry/DataDog for production monitoring

## 📝 Commands for Testing

```bash
# Typecheck after changes
pnpm typecheck

# Format updated files
pnpm prettier --write 'src/**/*.ts'

# Test API error responses
curl http://localhost:3001/api/posts/invalid-id
curl http://localhost:3001/api/notifications -H "Authorization: invalid"
```

---

**Migration Strategy:** Incremental, non-breaking changes that improve consistency and observability across the entire backend.
