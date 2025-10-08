# Backend Refactoring Opportunities - Library Project

**Analysis Date:** October 8, 2025
**Analyzed By:** AI Assistant

## Executive Summary

The library project is a well-structured Next.js 15 application with a modern tech stack (Prisma ORM, NextAuth, BullMQ, React Query). However, there are several refactoring opportunities that could improve code maintainability, consistency, performance, and developer experience.

**Priority Levels:**

- 🔴 **High Priority**: Issues that affect performance, maintainability, or could cause bugs
- 🟡 **Medium Priority**: Improvements that would enhance code quality and developer experience
- 🟢 **Low Priority**: Nice-to-have improvements

---

## 1. 🔴 Duplicate Prisma Client Instances

### Issue

Two separate Prisma client initialization files exist:

- `src/lib/db/prisma.ts` (used in 50+ files)
- `src/lib/db/index.ts` (exports `db`, appears unused)

### Impact

- Confusion about which import to use
- Potential for connection pool issues
- Inconsistent patterns across codebase

### Recommendation

1. Consolidate to a single Prisma client export
2. Choose one naming convention (`prisma` or `db`)
3. Update all imports to use the consolidated version

```typescript
// Recommended: src/lib/db.ts
import { PrismaClient } from "@prisma/client";

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log: process.env.NODE_ENV === "development" ? ["error", "warn"] : ["error"],
  });

if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}
```

---

## 2. 🟡 API Routes vs Server Actions Redundancy

### Issue

The project uses both API routes and Server Actions, with some functionality duplicated:

**API Routes (`src/app/api/`):**

- 24 route files
- Traditional REST-style endpoints
- Used for external/client-side API calls

**Server Actions (`src/*/actions/`):**

- Multiple action files per domain
- Used for form submissions and mutations
- Better integration with React 19/Next.js 15

### Current Pattern Examples

**Redundant Pattern:**

```typescript
// API Route: src/app/api/posts/route.ts
export async function GET(request: NextRequest) {
  const response = await listPosts({ limit, offset });
  return NextResponse.json(response);
}

// Server Action: Could do the same directly
```

### Recommendation

**Strategy: Server Actions First, API Routes for Specific Cases**

1. **Use Server Actions for:**
   - Form submissions
   - Mutations (create, update, delete)
   - Server-side data fetching in Server Components
   - Direct React integration

2. **Keep API Routes only for:**
   - External API consumption (webhooks, integrations)
   - File uploads/downloads
   - WebSocket/streaming endpoints
   - OAuth callbacks
   - Third-party integrations (Discord, etc.)

3. **Routes to Consider Removing/Migrating:**
   - `api/posts/route.ts` → Use server actions
   - `api/notifications/route.ts` → Use server actions
   - `api/mentions/search/route.ts` → Could be server action

4. **Routes to Keep:**
   - `api/auth/[...nextauth]/route.ts` (NextAuth requirement)
   - `api/upload-image/route.ts` (file upload)
   - `api/download-pdf/route.ts` (file download)
   - `api/discord/*` (external integration)
   - `api/admin/*` (external API access)

---

## 3. 🟡 Data Access Layer Inconsistencies

### Issue

Data access patterns vary across domains:

**Inconsistent Organization:**

```
src/posts/data/        ✓ Well-organized
src/authors/actions/   ✗ Missing data layer
src/groups/actions/    ✗ Mixing data access with actions
src/institutions/actions/ ✗ Direct Prisma calls in actions
```

### Current Problems

1. **Business logic mixed with data access:**

```typescript
// src/institutions/actions/joinInstitutionAction.ts
export const joinInstitutionAction = actionClient
  .action(async ({ parsedInput: input }) => {
    // Direct Prisma calls mixed with business logic
    const institution = await prisma.institution.findUnique({...});
    const existingMembership = await prisma.userInstitution.findUnique({...});

    // Business logic
    const userEmail = session.user.email;
    let domainAutoApproved = false;
    if (userEmail && institution.domain) {
      const userDomain = userEmail.split("@")[1];
      // ...validation logic
    }

    // More Prisma calls
    await prisma.userInstitution.create({...});
  });
```

2. **Inconsistent DTO patterns:**

- Some use explicit DTOs (`FeedPostDTO`, `InstitutionDTO`)
- Others return raw Prisma types
- Transformation logic scattered

### Recommendation

**Implement Consistent 3-Layer Architecture:**

```
┌─────────────────────┐
│  Actions/Routes     │  ← Handles requests, validation, auth
│  (presentation)     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Services           │  ← Business logic, orchestration
│  (business logic)   │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Data Access        │  ← Database queries, DTOs
│  (persistence)      │
└─────────────────────┘
```

**Example Refactoring:**

```typescript
// src/institutions/data/institutions.ts
export class InstitutionRepository {
  async findById(id: string) {
    return prisma.institution.findUnique({ where: { id } });
  }

  async findMembership(userId: string, institutionId: string) {
    return prisma.userInstitution.findUnique({
      where: { userId_institutionId: { userId, institutionId } },
    });
  }

  async createMembership(data: CreateMembershipInput) {
    return prisma.userInstitution.create({ data });
  }
}

// src/institutions/services/membership-service.ts
export class MembershipService {
  constructor(private repo: InstitutionRepository) {}

  async joinInstitution(
    userId: string,
    institutionId: string,
    userEmail: string
  ) {
    const institution = await this.repo.findById(institutionId);
    if (!institution) throw new Error("Institution not found");

    const existing = await this.repo.findMembership(userId, institutionId);
    if (existing) {
      return this.handleExistingMembership(existing);
    }

    const autoApproved = this.checkDomainMatch(userEmail, institution.domain);
    const status = autoApproved ? "APPROVED" : "PENDING";

    return this.repo.createMembership({ userId, institutionId, status });
  }

  private checkDomainMatch(
    email: string,
    institutionDomain: string | null
  ): boolean {
    if (!institutionDomain) return false;
    const userDomain = email.split("@")[1];
    return userDomain?.toLowerCase() === institutionDomain.toLowerCase();
  }
}

// src/institutions/actions/joinInstitutionAction.ts
export const joinInstitutionAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const session = await getSessionOrRedirect();

    const membershipService = new MembershipService(
      new InstitutionRepository()
    );
    const result = await membershipService.joinInstitution(
      session.user.id,
      input.institutionId,
      session.user.email
    );

    revalidatePath("/institutions");
    return result;
  });
```

---

## 4. 🟡 Background Job Worker Patterns

### Issue

Workers have duplicated patterns and could be more maintainable:

**Duplicated Code in Workers:**

- Event handler setup repeated
- Error handling patterns duplicated
- Shutdown logic similar across workers

### Current Pattern

```typescript
// Each worker repeats similar patterns
export class InstitutionWorker {
  private worker: Worker;

  constructor() {
    this.worker = new Worker("queue-name", async (job) => { ... });
    this.setupEventHandlers(); // Duplicated
  }

  private setupEventHandlers(): void {
    this.worker.on("completed", (job) => { /* similar code */ });
    this.worker.on("failed", (job, err) => { /* similar code */ });
    this.worker.on("error", (err) => { /* similar code */ });
  }

  async shutdown(): Promise<void> { /* similar code */ }
}
```

### Recommendation

**Create Abstract Base Worker Class:**

```typescript
// src/lib/workers/base-worker.ts
export abstract class BaseWorker<TJobData> {
  protected worker: Worker<TJobData>;
  protected abstract queueName: string;
  protected abstract concurrency: number;

  constructor() {
    this.worker = new Worker<TJobData>(
      this.queueName,
      async (job: Job<TJobData>) => {
        return await this.processJob(job);
      },
      {
        connection: redisConfig,
        concurrency: this.concurrency,
        ...this.getWorkerOptions(),
      }
    );

    this.setupEventHandlers();
  }

  protected abstract processJob(job: Job<TJobData>): Promise<void>;

  protected getWorkerOptions(): Partial<WorkerOptions> {
    return {};
  }

  private setupEventHandlers(): void {
    this.worker.on("completed", (job) => {
      this.onJobCompleted(job);
    });

    this.worker.on("failed", (job, err) => {
      this.onJobFailed(job, err);
    });

    this.worker.on("error", (err) => {
      this.onError(err);
    });
  }

  protected onJobCompleted(job: Job): void {
    console.log(`✅ ${this.queueName} job ${job.id} completed`);
  }

  protected onJobFailed(job: Job | undefined, error: Error): void {
    console.error(`❌ ${this.queueName} job ${job?.id} failed:`, error.message);
  }

  protected onError(error: Error): void {
    console.error(`🚨 ${this.queueName} worker error:`, error);
  }

  async shutdown(): Promise<void> {
    console.log(`🛑 Shutting down ${this.queueName} worker...`);
    await this.worker.close();
  }
}

// Usage example
export class InstitutionWorker extends BaseWorker<
  BackgroundJobs["extract-institutions"]
> {
  protected queueName = "institution-extraction";
  protected concurrency = 2;

  protected async processJob(
    job: Job<BackgroundJobs["extract-institutions"]>
  ): Promise<void> {
    const { paperId, arxivUrl } = job.data;
    console.log(`Processing institution extraction for paper ${paperId}`);

    // Actual processing logic
    await extractInstitutionsFromPaper(paperId, arxivUrl);
  }

  protected getWorkerOptions() {
    return {
      limiter: {
        max: 10,
        duration: 60000,
      },
    };
  }
}
```

---

## 5. 🔴 Notification System Complexity

### Issue

The notification system has multiple layers with complex flow:

```
Action → createNotification → queueExternalNotifications → JobQueue
  → ExternalNotificationWorker → [Email/Discord]Service → NotificationDelivery
```

**Problems:**

1. Notification preferences checked in multiple places
2. Delivery tracking adds complexity
3. Error handling spread across layers
4. Difficult to test and debug

### Current Flow

```typescript
// notifications.ts
export async function createNotification(params) {
  const notification = await prisma.notification.create({ data: params });
  await queueExternalNotifications(notification.id, notification.userId, notification.type);
  return notification;
}

async function queueExternalNotifications(notificationId, userId, type) {
  const enabledChannels = await getEnabledChannels(userId, type);
  await JobQueueService.queueExternalNotification(...);
}

// external-notification-worker.ts
private async processSendNotification(job) {
  const { notificationId, channels, userId } = job.data;
  const notification = await prisma.notification.findUnique({...}); // Re-fetch
  const preferences = await this.getUserPreferences(userId, notification.type); // Re-check
  // ... send to channels
}
```

### Recommendation

**Simplify Notification Architecture:**

```typescript
// src/lib/notifications/notification-service.ts
export class NotificationService {
  constructor(
    private repo: NotificationRepository,
    private channels: NotificationChannel[]
  ) {}

  async send(notification: NotificationInput): Promise<void> {
    // 1. Create notification record
    const saved = await this.repo.create(notification);

    // 2. Get user preferences once
    const preferences = await this.repo.getPreferences(
      notification.userId,
      notification.type
    );

    // 3. Send to enabled channels
    const sendPromises = this.channels
      .filter((channel) => preferences[channel.name])
      .map((channel) => channel.send(saved));

    // 4. Track results
    const results = await Promise.allSettled(sendPromises);
    await this.repo.recordDeliveries(saved.id, results);
  }
}

// Usage
const notificationService = new NotificationService(
  new NotificationRepository(),
  [new EmailChannel(), new DiscordChannel()]
);

await notificationService.send({
  userId: "user-id",
  type: "MENTION",
  title: "You were mentioned",
  message: "...",
});
```

**Benefits:**

- Single responsibility per class
- Easier to test (mock channels)
- Clear data flow
- Reduced database queries

---

## 6. 🟡 Query Optimization Opportunities

### Issue

Some data access patterns could lead to N+1 queries or inefficient fetches.

**Example from `loadFeedPosts`:**

```typescript
// Current: Fetches all posts, then fetches quoted posts separately
const allRows = await prisma.post.findMany({ ... });
const allQuotedPostIds = [...new Set(allRows.flatMap(row => row.quotedPostIds))];

if (allQuotedPostIds.length > 0) {
  const quotedPosts = await prisma.post.findMany({
    where: { id: { in: allQuotedPostIds } }
  });
}
```

**Issue from `loadInstitutions`:**

```typescript
// Loads all institutions with papers with authors - could be heavy
const institutions = await prisma.institution.findMany({
  include: {
    papers: {
      include: {
        paper: {
          include: {
            paperAuthors: { include: { author: true } },
            userPaperInteractions: { where: { starred: true } },
          },
        },
      },
    },
  },
});
```

### Recommendations

1. **Use Prisma Nested Includes Efficiently:**

```typescript
// Include quoted posts in the initial query
const posts = await prisma.post.findMany({
  include: {
    author: true,
    paper: { ... },
    quotedPosts: {  // Direct relation
      include: {
        author: { select: { id: true, name: true, email: true, image: true } }
      }
    }
  }
});
```

2. **Add Database Indices:**

```prisma
// prisma/schema.prisma
model Post {
  // ...

  @@index([createdAt, id])
  @@index([paperId])
  @@index([authorId])
}

model UserPaperInteraction {
  // ...

  @@index([userId, starred])
  @@index([paperId, starred])
}

model Notification {
  // ...

  @@index([userId, isRead, createdAt])
}
```

3. **Implement Cursor-Based Pagination Properly:**

```typescript
export async function loadFeedPosts({ limit = 10, cursor }: PaginationParams) {
  const posts = await prisma.post.findMany({
    take: limit + 1,
    ...(cursor && {
      cursor: { id: cursor },
      skip: 1,
    }),
    orderBy: [{ createdAt: "desc" }, { id: "desc" }],
    include: {
      /* ... */
    },
  });

  const hasMore = posts.length > limit;
  const items = hasMore ? posts.slice(0, -1) : posts;
  const nextCursor = hasMore ? items[items.length - 1].id : undefined;

  return { items, nextCursor, hasMore };
}
```

4. **Use Prisma Raw Queries for Complex Aggregations:**

```typescript
// Instead of loading all data and aggregating in memory
export async function getInstitutionStats() {
  const stats = await prisma.$queryRaw`
    SELECT
      i.id,
      i.name,
      COUNT(DISTINCT pi.paper_id) as paper_count,
      COUNT(DISTINCT pa.author_id) as author_count,
      AVG(COALESCE((
        SELECT COUNT(*) FROM user_paper_interaction upi
        WHERE upi.paper_id = pi.paper_id AND upi.starred = true
      ), 0)) as avg_stars
    FROM institution i
    LEFT JOIN paper_institution pi ON pi.institution_id = i.id
    LEFT JOIN paper_author pa ON pa.paper_id = pi.paper_id
    GROUP BY i.id, i.name
    ORDER BY paper_count DESC
  `;

  return stats;
}
```

---

## 7. 🟢 Type Safety Improvements

### Issue

Mixed usage of Prisma-generated types and custom DTOs can lead to type inconsistencies.

**Current Pattern:**

```typescript
// Sometimes uses Prisma types
import type { Post, Paper, User } from "@prisma/client";

// Sometimes uses custom DTOs
export type FeedPostDTO = { ... };

// Sometimes uses type assertions
return toFeedPostDTO(
  row as unknown as PrismaPost,  // Type assertion needed
  usersMap,
  papersMap
);
```

### Recommendations

1. **Define Clear Type Hierarchy:**

```typescript
// src/types/domain.ts

// Domain types (what the application uses)
export type Post = {
  id: string;
  title: string;
  content: string | null;
  author: Author;
  paper?: Paper;
  createdAt: Date;
};

// Database types (re-export from Prisma with alias)
export type PostEntity = PrismaPost;
export type PaperEntity = PrismaPaper;

// API types (what gets serialized over the wire)
export type PostDTO = {
  id: string;
  title: string;
  content: string | null;
  author: AuthorDTO;
  paper?: PaperDTO;
  createdAt: string; // Date serialized as string
};
```

2. **Use Zod for Runtime Validation:**

```typescript
import { z } from "zod";

// Define schema
export const PostSchema = z.object({
  id: z.string().cuid(),
  title: z.string().min(1).max(255),
  content: z.string().nullable(),
  authorId: z.string().cuid(),
});

// Infer types
export type Post = z.infer<typeof PostSchema>;

// Use in API routes
export async function POST(request: Request) {
  const body = await request.json();
  const post = PostSchema.parse(body); // Runtime validation + type safety
  // ...
}
```

3. **Create Type-Safe Query Builders:**

```typescript
// src/lib/query-builders.ts
export const postSelect = {
  id: true,
  title: true,
  content: true,
  author: {
    select: {
      id: true,
      name: true,
      email: true,
    },
  },
} as const satisfies Prisma.PostSelect;

// Usage ensures type safety
const post = await prisma.post.findUnique({
  where: { id },
  select: postSelect,
});
// Type of `post` is automatically inferred correctly
```

---

## 8. 🟡 Error Handling Standardization

### Issue

Error handling varies across the codebase:

**Different Patterns:**

```typescript
// Pattern 1: Try-catch with console.error
try {
  const result = await somethingRisky();
} catch (error) {
  console.error("Error:", error);
  throw new Error("Failed to process");
}

// Pattern 2: ActionError from actionClient
if (invalid) {
  throw new ActionError("Invalid input");
}

// Pattern 3: Return error objects
return { error: "Something went wrong" };

// Pattern 4: NextResponse with error
return NextResponse.json({ error: "Not found" }, { status: 404 });
```

### Recommendations

1. **Create Error Hierarchy:**

```typescript
// src/lib/errors.ts
export class AppError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number = 500,
    public details?: unknown
  ) {
    super(message);
    this.name = this.constructor.name;
  }
}

export class ValidationError extends AppError {
  constructor(message: string, details?: unknown) {
    super(message, "VALIDATION_ERROR", 400, details);
  }
}

export class NotFoundError extends AppError {
  constructor(resource: string, id?: string) {
    super(
      `${resource}${id ? ` with id ${id}` : ""} not found`,
      "NOT_FOUND",
      404
    );
  }
}

export class AuthorizationError extends AppError {
  constructor(message: string = "Unauthorized") {
    super(message, "UNAUTHORIZED", 403);
  }
}

export class ConflictError extends AppError {
  constructor(message: string, details?: unknown) {
    super(message, "CONFLICT", 409, details);
  }
}
```

2. **Centralized Error Handler for API Routes:**

```typescript
// src/lib/api/error-handler.ts
export function handleApiError(error: unknown): NextResponse {
  if (error instanceof AppError) {
    return NextResponse.json(
      {
        error: error.message,
        code: error.code,
        details: error.details,
      },
      { status: error.statusCode }
    );
  }

  if (error instanceof z.ZodError) {
    return NextResponse.json(
      {
        error: "Validation failed",
        code: "VALIDATION_ERROR",
        details: error.errors,
      },
      { status: 400 }
    );
  }

  // Log unexpected errors
  console.error("Unexpected error:", error);

  return NextResponse.json(
    {
      error: "Internal server error",
      code: "INTERNAL_ERROR",
    },
    { status: 500 }
  );
}

// Usage in API routes
export async function GET(request: NextRequest) {
  try {
    const data = await fetchData();
    return NextResponse.json(data);
  } catch (error) {
    return handleApiError(error);
  }
}
```

3. **Error Logging Service:**

```typescript
// src/lib/logging/logger.ts
export class Logger {
  static error(
    message: string,
    error: Error,
    context?: Record<string, unknown>
  ) {
    const errorLog = {
      timestamp: new Date().toISOString(),
      message,
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack,
      },
      context,
    };

    // Log to console in development
    if (process.env.NODE_ENV === "development") {
      console.error(JSON.stringify(errorLog, null, 2));
    }

    // Send to error tracking service in production
    // e.g., Sentry, DataDog, etc.
  }

  static info(message: string, data?: Record<string, unknown>) {
    // ...
  }
}
```

---

## 9. 🟢 Configuration Management

### Issue

Configuration scattered across files:

- Environment variables accessed directly
- Magic values hardcoded
- No type safety for config

### Recommendation

**Centralized Configuration:**

```typescript
// src/lib/config.ts
import { z } from "zod";

const configSchema = z.object({
  // Database
  database: z.object({
    url: z.string().url(),
  }),

  // Redis
  redis: z.object({
    host: z.string(),
    port: z.coerce.number(),
    password: z.string().optional(),
    tls: z.coerce.boolean().default(false),
  }),

  // Auth
  auth: z.object({
    secret: z.string().min(32),
    url: z.string().url(),
    allowedDomains: z.string().transform((s) => s.split(",")),
  }),

  // External Services
  services: z.object({
    discord: z.object({
      clientId: z.string().optional(),
      clientSecret: z.string().optional(),
    }),
    anthropic: z.object({
      apiKey: z.string().optional(),
    }),
    aws: z.object({
      region: z.string().default("us-east-1"),
      s3Bucket: z.string().optional(),
    }),
  }),

  // Features
  features: z.object({
    emailNotifications: z.coerce.boolean().default(false),
    devMode: z.coerce.boolean().default(false),
  }),
});

type Config = z.infer<typeof configSchema>;

function loadConfig(): Config {
  try {
    return configSchema.parse({
      database: {
        url: process.env.DATABASE_URL,
      },
      redis: {
        host: process.env.REDIS_HOST,
        port: process.env.REDIS_PORT,
        password: process.env.REDIS_PASSWORD,
        tls: process.env.REDIS_TLS,
      },
      auth: {
        secret: process.env.NEXTAUTH_SECRET,
        url: process.env.NEXTAUTH_URL,
        allowedDomains: process.env.ALLOWED_EMAIL_DOMAINS,
      },
      services: {
        discord: {
          clientId: process.env.DISCORD_CLIENT_ID,
          clientSecret: process.env.DISCORD_CLIENT_SECRET,
        },
        anthropic: {
          apiKey: process.env.ANTHROPIC_API_KEY,
        },
        aws: {
          region: process.env.AWS_REGION,
          s3Bucket: process.env.AWS_S3_BUCKET,
        },
      },
      features: {
        emailNotifications: process.env.ENABLE_EMAIL_NOTIFICATIONS,
        devMode: process.env.DEV_MODE,
      },
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error("❌ Configuration validation failed:");
      error.errors.forEach((err) => {
        console.error(`  - ${err.path.join(".")}: ${err.message}`);
      });
    }
    throw new Error("Failed to load configuration");
  }
}

export const config = loadConfig();

// Usage
import { config } from "@/lib/config";

const redisConnection = {
  host: config.redis.host,
  port: config.redis.port,
};
```

---

## 10. 🟡 Testing Infrastructure

### Issue

No visible test files in the `src` directory, which makes refactoring risky.

### Recommendations

1. **Set Up Testing Framework:**

```bash
pnpm add -D vitest @testing-library/react @testing-library/jest-dom
```

2. **Test Configuration:**

```typescript
// vitest.config.ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
```

3. **Example Tests:**

```typescript
// src/lib/notifications/__tests__/notification-service.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { NotificationService } from "../notification-service";

describe("NotificationService", () => {
  let service: NotificationService;
  let mockRepo: any;
  let mockChannels: any[];

  beforeEach(() => {
    mockRepo = {
      create: vi.fn(),
      getPreferences: vi.fn(),
      recordDeliveries: vi.fn(),
    };

    mockChannels = [
      {
        name: "email",
        send: vi.fn(),
      },
      {
        name: "discord",
        send: vi.fn(),
      },
    ];

    service = new NotificationService(mockRepo, mockChannels);
  });

  it("should create notification and send to enabled channels", async () => {
    const notification = {
      userId: "user-1",
      type: "MENTION",
      title: "Test",
      message: "Test message",
    };

    mockRepo.create.mockResolvedValue({ id: "notif-1", ...notification });
    mockRepo.getPreferences.mockResolvedValue({
      email: true,
      discord: false,
    });

    await service.send(notification);

    expect(mockRepo.create).toHaveBeenCalledWith(notification);
    expect(mockChannels[0].send).toHaveBeenCalled();
    expect(mockChannels[1].send).not.toHaveBeenCalled();
  });
});
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1-2)

1. ✓ Consolidate Prisma client instances
2. ✓ Set up testing infrastructure
3. ✓ Implement error hierarchy
4. ✓ Create configuration management

### Phase 2: Architecture (Week 3-4)

1. ✓ Refactor data access layer (start with posts)
2. ✓ Create base worker class
3. ✓ Simplify notification system
4. ✓ Add database indices

### Phase 3: Cleanup (Week 5-6)

1. ✓ Remove redundant API routes
2. ✓ Standardize error handling
3. ✓ Improve type safety
4. ✓ Add unit tests for critical paths

### Phase 4: Optimization (Week 7-8)

1. ✓ Optimize queries
2. ✓ Implement proper pagination
3. ✓ Add integration tests
4. ✓ Performance profiling

---

## Metrics & Success Criteria

### Code Quality Metrics

- Reduce code duplication by 30%
- Achieve 70%+ test coverage
- Zero linting/type errors
- Consistent patterns across all domains

### Performance Metrics

- Reduce database queries per request by 25%
- Feed load time < 500ms (p95)
- Background job processing time < 2s (p95)

### Developer Experience

- Clear separation of concerns
- Easy to add new features
- Self-documenting code structure
- Comprehensive error messages

---

## Additional Resources

### Recommended Reading

- [Prisma Best Practices](https://www.prisma.io/docs/guides/performance-and-optimization)
- [Next.js Server Actions](https://nextjs.org/docs/app/building-your-application/data-fetching/server-actions-and-mutations)
- [BullMQ Patterns](https://docs.bullmq.io/patterns/introduction)
- [Clean Architecture in TypeScript](https://khalilstemmler.com/articles/software-design-architecture/organizing-app-logic/)

### Tools

- Prisma Studio for database inspection
- React Query Devtools for client state
- BullMQ Board for job monitoring
- Sentry/DataDog for error tracking

---

## Conclusion

The library project has a solid foundation but would benefit significantly from these refactoring efforts. The recommendations focus on:

1. **Consistency** - Standardized patterns across the codebase
2. **Maintainability** - Clear separation of concerns
3. **Performance** - Optimized queries and reduced redundancy
4. **Developer Experience** - Better types, errors, and testability

Implementing these changes incrementally will improve code quality without disrupting ongoing development.

---

**Document Version:** 1.0
**Last Updated:** October 8, 2025
