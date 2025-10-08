# Error Handling Migration Status

**Created:** October 8, 2025

## ✅ Completed

### Core Infrastructure
- ✅ Error class hierarchy (`src/lib/errors.ts`)
- ✅ API error handler (`src/lib/api/error-handler.ts`)
- ✅ Logger service (`src/lib/logging/logger.ts`)
- ✅ Updated `actionClient.ts` to use new errors
- ✅ Created centralized API exports

### API Routes Updated (8/22)
- ✅ `/api/notifications` 
- ✅ `/api/notifications/count`
- ✅ `/api/posts/[id]`
- ✅ `/api/posts` 
- ✅ `/api/papers/[postId]/data`
- ✅ `/api/institutions`
- ✅ `/api/authors`
- ✅ `/api/mentions/search`

## 🚧 Remaining Work

### API Routes to Update (14 remaining)
1. `/api/notification-preferences/route.ts`
2. `/api/notifications/mark-read/route.ts`
3. `/api/papers/[postId]/institutions/route.ts`
4. `/api/institutions/[name]/route.ts`
5. `/api/authors/[authorId]/route.ts`
6. `/api/admin/institutions/route.ts`
7. `/api/admin/users/route.ts`
8. `/api/discord/auth/route.ts`
9. `/api/discord/link/route.ts`
10. `/api/discord/test/route.ts`
11. `/api/email-test/route.ts`
12. `/api/upload-image/route.ts`
13. `/api/analyze-pdf/route.ts`
14. `/api/chat/library-bot/route.ts`
15. `/api/download-pdf/route.ts`

### console.error Replacement (201 instances across 66 files)

**Priority Areas:**
- `/lib/**` backend services (high priority)
- `/workers/**` background jobs (high priority)  
- `/posts/actions/**` server actions (high priority)
- Components (lower priority - most are client-side)

**Backend Files Needing Logger:**
- `lib/workers/*.ts` (5 files)
- `lib/external-notifications/*.ts` (2 files)
- `lib/arxiv-auto-import.ts`
- `lib/notifications.ts`
- `lib/mention-resolution.ts`
- `lib/paper-abstract-service.ts`
- `lib/auto-tagging-service.ts`
- `posts/actions/*.ts` (3 files)

## 📋 Migration Pattern

### For Simple API Routes
```typescript
import { withErrorHandler } from "@/lib/api/error-handler";

export const GET = withErrorHandler(async (request) => {
  // Handler code - errors auto-caught
  return NextResponse.json(data);
});
```

### For Complex API Routes  
```typescript
import { handleApiError, AuthenticationError } from "@/lib/errors";

export async function GET(request) {
  try {
    if (!authenticated) {
      throw new AuthenticationError();
    }
    // ... handler code
    return NextResponse.json(data);
  } catch (error) {
    return handleApiError(error, { endpoint: "GET /api/..." });
  }
}
```

### For Backend Services
```typescript
import { Logger } from "@/lib/logging/logger";

// Replace:
console.error("Error message:", error);

// With:
Logger.error("Error message", error, { context: "additional info" });
```

## 🎯 Next Steps

1. **Complete remaining API routes** - Use patterns above
2. **Replace console.error in backend** - Systematic file-by-file  
3. **Update server actions** - Use typed errors where appropriate
4. **Test error scenarios** - Verify proper error responses
5. **Monitor in production** - Set up error tracking integration

## 📊 Progress
- Infrastructure: 100% ✅
- API Routes: 36% (8/22)
- Backend logging: 0% (0/201 console.error replaced)
- Server Actions: 0%

**Overall: ~15% complete**

