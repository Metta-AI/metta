# Architecture Comparison - Current vs Proposed

## Current Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                    │
├──────────────────────┬──────────────────────────────────┤
│   Server Actions     │        API Routes                │
│   (Form handlers)    │        (REST endpoints)          │
│                      │                                  │
│   • Mixed logic      │   • Thin handlers               │
│   • Direct DB calls  │   • Call data functions         │
│   • Some use actions │   • JSON responses              │
│     client pattern   │                                  │
└──────────────────────┴──────────────────────────────────┘
                    │                │
                    └────────┬───────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│              DATA ACCESS LAYER (Inconsistent)            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  src/posts/data/                                        │
│  ├── feed.ts              ✓ Well organized             │
│  ├── papers.ts            ✓ Has DTOs                   │
│  └── post.ts              ✓ Transformation functions   │
│                                                          │
│  src/institutions/actions/                              │
│  ├── joinInstitutionAction.ts  ✗ Mixed logic          │
│  ├── createInstitutionAction.ts ✗ Direct Prisma       │
│  └── ...                        ✗ No separation        │
│                                                          │
│  src/groups/actions/                                    │
│  └── ...                        ✗ Similar issues       │
│                                                          │
└──────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    DATABASE (Prisma)                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ⚠️  TWO client instances:                              │
│     • src/lib/db/prisma.ts (used everywhere)           │
│     • src/lib/db/index.ts (exports 'db', rarely used)  │
│                                                          │
│  ⚠️  Missing indices on:                                │
│     • Post (createdAt, paperId, authorId)              │
│     • Notification (userId, isRead, createdAt)         │
│     • UserPaperInteraction (starred, queued)           │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              BACKGROUND JOBS (BullMQ/Redis)              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Workers (All follow similar patterns):                 │
│  ├── InstitutionWorker        ⚠️ Duplicated setup     │
│  ├── LLMWorker                ⚠️ Duplicated handlers   │
│  ├── TaggingWorker            ⚠️ Duplicated shutdown   │
│  └── ExternalNotificationWorker ⚠️ Complex flow       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                    │
├──────────────────────┬──────────────────────────────────┤
│   Server Actions     │   API Routes (Minimal)           │
│   (Primary Interface)│   (Special cases only)           │
│                      │                                  │
│   ✓ Use for:         │   ✓ Keep only for:              │
│   • Forms            │   • File uploads/downloads       │
│   • Mutations        │   • OAuth callbacks              │
│   • Server-side data │   • Webhooks                     │
│                      │   • External integrations        │
│   ✓ Thin orchestration│                                 │
│   ✓ Call services    │   ✓ Use error handler           │
└──────────────────────┴──────────────────────────────────┘
                    │                │
                    └────────┬───────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│                   SERVICE LAYER (NEW)                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Business Logic & Orchestration:                        │
│                                                          │
│  src/posts/services/                                    │
│  └── post-service.ts          • Create posts           │
│                                • Process arXiv imports  │
│                                • Handle mentions        │
│                                                          │
│  src/institutions/services/                             │
│  └── membership-service.ts    • Join institution       │
│                                • Domain validation      │
│                                • Approval logic         │
│                                                          │
│  src/notifications/                                     │
│  └── notification-service.ts  • Create notifications   │
│                                • Send to channels       │
│                                • Track delivery         │
│                                                          │
│  src/papers/services/                                   │
│  └── paper-service.ts         • Import papers          │
│                                • Enrich metadata        │
│                                • Queue jobs             │
│                                                          │
└──────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                DATA ACCESS LAYER (Repositories)          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Data Queries & Transformations:                        │
│                                                          │
│  src/posts/data/                                        │
│  ├── post-repository.ts       • findById              │
│  │                             • findMany with filters │
│  │                             • create/update/delete  │
│  └── post-dto.ts              • toPostDTO             │
│                                • Type definitions      │
│                                                          │
│  src/institutions/data/                                 │
│  ├── institution-repository.ts • findById              │
│  │                             • findMembership        │
│  │                             • createMembership      │
│  └── institution-dto.ts       • toInstitutionDTO      │
│                                                          │
│  src/notifications/data/                                │
│  ├── notification-repository.ts • create              │
│  │                              • getPreferences       │
│  │                              • recordDeliveries     │
│  └── notification-dto.ts        • Type definitions    │
│                                                          │
│  ✓ All repositories extend BaseRepository              │
│  ✓ Consistent query patterns                           │
│  ✓ Type-safe DTOs                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    DATABASE (Prisma)                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ✓ Single client instance:                             │
│    src/lib/db.ts                                       │
│                                                          │
│  ✓ Optimized with indices:                             │
│    • Post (createdAt, paperId, authorId)              │
│    • Notification (userId, isRead, createdAt)         │
│    • UserPaperInteraction (starred, queued)           │
│    • Comment (postId, authorId, createdAt)            │
│                                                          │
│  ✓ Connection pooling configured                       │
│  ✓ Query logging in development                        │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              BACKGROUND JOBS (BullMQ/Redis)              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ✓ All workers extend BaseWorker:                      │
│                                                          │
│  src/lib/workers/                                       │
│  ├── base-worker.ts           • Common setup           │
│  │                             • Event handlers         │
│  │                             • Graceful shutdown      │
│  │                                                      │
│  ├── institution-worker.ts    • Extends BaseWorker    │
│  ├── llm-worker.ts            • Extends BaseWorker    │
│  ├── tagging-worker.ts        • Extends BaseWorker    │
│  └── notification-worker.ts   • Extends BaseWorker    │
│                                • Simplified flow        │
│                                                          │
│  ✓ Consistent patterns                                 │
│  ✓ Better error handling                               │
│  ✓ Easier to test                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Cross-Cutting Concerns

### Current State

```
Error Handling:
  ⚠️ Inconsistent patterns
  ⚠️ console.error scattered
  ⚠️ Different error types
  ⚠️ No centralized logging

Configuration:
  ⚠️ process.env everywhere
  ⚠️ No validation
  ⚠️ Magic values

Type Safety:
  ⚠️ Mixed Prisma/custom types
  ⚠️ Type assertions needed
  ⚠️ DTOs not standardized

Testing:
  ⚠️ No test infrastructure
  ⚠️ Hard to test (mixed concerns)
  ⚠️ No mocking strategy
```

### Proposed State

```
Error Handling:
  ✓ Centralized error hierarchy
  ✓ Consistent error responses
  ✓ Structured logging
  ✓ Error tracking integration

Configuration:
  ✓ Centralized config service
  ✓ Type-safe with Zod validation
  ✓ Environment-aware defaults
  ✓ Clear documentation

Type Safety:
  ✓ Clear type hierarchy
  ✓ Entity vs DTO separation
  ✓ Zod for runtime validation
  ✓ No type assertions needed

Testing:
  ✓ Vitest infrastructure
  ✓ Testable services
  ✓ Mock repositories
  ✓ >70% coverage target
```

---

## Data Flow Comparison

### Current: Creating a Post with Paper Import

```
User submits form
    │
    ▼
createPostAction
    │
    ├─ Validate session
    ├─ Check ban status
    ├─ Parse images/mentions/quotedPosts
    ├─ Detect arXiv URL
    ├─ processArxivAutoImport()  ← Direct business logic
    │   ├─ Extract arXiv ID
    │   ├─ Fetch arXiv API
    │   ├─ Check if paper exists (Prisma query)
    │   ├─ Create paper (Prisma query)
    │   └─ Create authors (Prisma queries)
    │
    ├─ Create post (Prisma query)
    ├─ queueArxivInstitutionProcessing()
    ├─ resolveMentions() (Prisma queries)
    └─ createMentionNotifications()
        ├─ Create notifications (Prisma queries)
        └─ queueExternalNotifications()

⚠️ Issues:
  • 150+ lines in single function
  • Mixed concerns (validation, business logic, data access)
  • Hard to test
  • Hard to reuse logic
  • Multiple database calls
```

### Proposed: Creating a Post with Paper Import

```
User submits form
    │
    ▼
createPostAction
    │
    ├─ Validate session & input (thin orchestration)
    └─ postService.createPost(data)
            │
            ▼
        PostService
            │
            ├─ paperService.importFromArxiv(url)
            │       │
            │       ├─ paperRepository.findByExternalId()
            │       └─ paperRepository.create()
            │
            ├─ postRepository.create(data)
            │
            ├─ mentionService.resolveMentions(mentions)
            │       │
            │       └─ userRepository.findByMentions()
            │
            ├─ notificationService.createMentions(resolved)
            │       │
            │       ├─ notificationRepository.create()
            │       └─ Queue external notifications
            │
            └─ jobQueue.queueInstitutionExtraction()

✓ Benefits:
  • Thin actions (orchestration only)
  • Clear separation of concerns
  • Easy to test each service
  • Reusable services
  • Optimized database calls
```

---

## Notification System Flow

### Current Architecture

```
Action triggers notification
    │
    ▼
createNotification(params)
    │
    ├─ prisma.notification.create()
    │
    └─ queueExternalNotifications(id, userId, type)
            │
            ├─ getEnabledChannels(userId, type)  ← DB query
            │
            └─ JobQueue.queueExternalNotification()
                    │
                    ▼
            ExternalNotificationWorker
                    │
                    ├─ prisma.notification.findUnique()  ← Re-fetch!
                    ├─ getUserPreferences(userId, type)  ← Re-fetch!
                    │
                    ├─ sendEmailNotification()
                    │   └─ emailService.sendNotification()
                    │       ├─ prisma.notificationDelivery.create()
                    │       └─ Send via AWS SES
                    │
                    └─ sendDiscordNotification()
                        └─ discordBot.sendNotification()
                            ├─ prisma.notificationDelivery.create()
                            └─ Send via Discord API

⚠️ Issues:
  • 4+ database queries for single notification
  • Re-fetching data multiple times
  • Complex nested callbacks
  • Hard to test
  • Error handling spread across layers
```

### Proposed Architecture

```
Action triggers notification
    │
    ▼
notificationService.send(params)
    │
    ├─ notificationRepository.create(params)
    │
    ├─ notificationRepository.getPreferences(userId, type)  ← Single query
    │
    ├─ channels.filter(enabled)
    │
    ├─ Promise.allSettled(
    │   channels.map(channel => channel.send(notification))
    │ )
    │
    └─ notificationRepository.recordDeliveries(results)

✓ Benefits:
  • 2 database queries total
  • Linear, easy-to-follow flow
  • Simple to test (mock channels)
  • Clear error handling
  • Channel abstraction
```

---

## File Organization Comparison

### Current Structure (Inconsistent)

```
src/
├── posts/
│   ├── actions/           ✓ Server actions
│   └── data/              ✓ Data access
│
├── institutions/
│   └── actions/           ✗ Mixed logic + data access
│
├── groups/
│   └── actions/           ✗ Mixed logic + data access
│
├── authors/
│   └── actions/           ✗ Mixed logic + data access
│
└── lib/
    ├── notifications.ts   ✓ Some service logic
    ├── auth.ts           ✓ Config
    └── ...               Mixed utilities
```

### Proposed Structure (Consistent)

```
src/
├── posts/
│   ├── actions/          ← Thin orchestration
│   ├── services/         ← Business logic (NEW)
│   ├── data/             ← Repository + DTOs
│   └── types/            ← Type definitions (NEW)
│
├── institutions/
│   ├── actions/          ← Thin orchestration
│   ├── services/         ← Business logic (NEW)
│   ├── data/             ← Repository + DTOs (NEW)
│   └── types/            ← Type definitions (NEW)
│
├── groups/
│   ├── actions/          ← Thin orchestration
│   ├── services/         ← Business logic (NEW)
│   ├── data/             ← Repository + DTOs (NEW)
│   └── types/            ← Type definitions (NEW)
│
├── notifications/
│   ├── services/         ← Unified notification service (NEW)
│   ├── channels/         ← Email, Discord abstractions (NEW)
│   ├── data/             ← Repository
│   └── types/            ← Type definitions
│
└── lib/
    ├── config.ts         ← Centralized config (NEW)
    ├── errors.ts         ← Error hierarchy (NEW)
    ├── logging/          ← Structured logging (NEW)
    └── workers/
        ├── base-worker.ts  ← Abstract base (NEW)
        └── ...workers      ← Extend base

All domains follow consistent pattern:
  actions/ → services/ → data/ → database
```

---

## Testing Strategy

### Current: Hard to Test

```typescript
// createPostAction.ts - 185 lines, hard to test

export const createPostAction = actionClient
  .action(async ({ parsedInput }) => {
    // Direct Prisma calls
    // Mixed business logic
    // Hard to mock
    // Many side effects
  });

❌ Problems:
  • Can't test in isolation
  • Need real database
  • Hard to mock dependencies
  • Side effects everywhere
```

### Proposed: Easy to Test

```typescript
// post-service.test.ts

describe("PostService", () => {
  let service: PostService;
  let mockRepo: MockRepository;
  let mockPaperService: MockPaperService;

  beforeEach(() => {
    mockRepo = new MockRepository();
    mockPaperService = new MockPaperService();
    service = new PostService(mockRepo, mockPaperService);
  });

  it("should create post with imported paper", async () => {
    mockPaperService.importFromArxiv.mockResolvedValue({ id: "paper-1" });
    mockRepo.create.mockResolvedValue({ id: "post-1" });

    const result = await service.createPost({
      title: "Test",
      content: "https://arxiv.org/abs/1234.5678"
    });

    expect(result.id).toBe("post-1");
    expect(mockPaperService.importFromArxiv).toHaveBeenCalled();
  });
});

✓ Benefits:
  • Test in isolation
  • Fast (no database)
  • Mock dependencies
  • Test edge cases
```

---

## Performance Comparison

### Database Queries: Load Feed (10 posts)

**Current:**

```
1. SELECT posts (with author, paper, comments)
2. SELECT quoted posts for all IDs
3. SELECT user interactions for all paper IDs
4. (In application) Sort by lastActivity, paginate

Total: 3 queries + in-memory processing
Time: ~200-400ms
```

**Proposed:**

```
1. SELECT posts (with author, paper, comments, quotedPosts)
   - Single query with proper JOINs
   - Cursor-based pagination
   - Filtered by index

2. SELECT user interactions for current page only
   - Scoped to current page papers

Total: 2 queries, no in-memory processing
Time: ~100-150ms (with indices)
```

### Background Job Processing

**Current:**

```
Institution Worker → Re-fetch paper → Extract → Update
Notification Worker → Re-fetch notification → Re-fetch preferences → Send

Multiple database round-trips per job
```

**Proposed:**

```
Institution Worker → Process with job data → Update once
Notification Worker → Process with job data → Send → Record once

Minimal database queries per job
```

---

## Migration Path

### Phase 1: Foundation (No Breaking Changes)

```
✓ Add indices (performance win)
✓ Fix Prisma duplication
✓ Add error classes
✓ Add config service
→ Existing code keeps working
```

### Phase 2: New Patterns (Parallel Development)

```
✓ Create service layer for posts (NEW)
✓ Create base worker (NEW)
✓ Keep old actions working
→ New features use new patterns
→ Old features still work
```

### Phase 3: Migration (Gradual)

```
✓ Migrate posts actions to use services
✓ Migrate institutions actions
✓ Migrate groups actions
→ One domain at a time
→ Test thoroughly
```

### Phase 4: Cleanup

```
✓ Remove redundant API routes
✓ Consolidate data access
✓ Achieve test coverage
→ Reap benefits!
```

---

## Key Benefits Summary

| Aspect                   | Current      | Proposed           | Improvement |
| ------------------------ | ------------ | ------------------ | ----------- |
| **Code Organization**    | Inconsistent | Consistent 3-layer | 🟢 High     |
| **Testability**          | Hard         | Easy               | 🟢 High     |
| **Performance**          | Good         | Better             | 🟡 Medium   |
| **Maintainability**      | Medium       | High               | 🟢 High     |
| **Type Safety**          | Good         | Excellent          | 🟡 Medium   |
| **Error Handling**       | Inconsistent | Standardized       | 🟢 High     |
| **Developer Experience** | Medium       | High               | 🟢 High     |
| **Database Queries**     | Sub-optimal  | Optimized          | 🟡 Medium   |
| **Code Duplication**     | Some         | Minimal            | 🟡 Medium   |

---

## Conclusion

The proposed architecture maintains all current functionality while providing:

1. **Better separation of concerns** - Clear layers with single responsibilities
2. **Improved testability** - Mock at service/repository boundaries
3. **Enhanced performance** - Fewer queries, better indices
4. **Consistent patterns** - Same structure across all domains
5. **Easier maintenance** - Changes isolated to appropriate layers

The migration can be done incrementally without breaking existing functionality.

---

**Document Version:** 1.0
**Last Updated:** October 8, 2025
