# Library Refactoring Guide

**Last Updated:** 2025-09-30
**Estimated Total Line Reduction:** 4,600+ lines (40-45% of UI layer)
**Estimated Timeline:** 6-8 weeks for complete refactor

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Critical Issues](#critical-issues)
- [Library Recommendations](#library-recommendations)
- [Line Reduction Strategies](#line-reduction-strategies)
- [Implementation Roadmap](#implementation-roadmap)
- [Quick Wins](#quick-wins)
- [Detailed Analysis](#detailed-analysis)

---

## Executive Summary

The library project has grown to 208 files with an 18,000 line PR indicating rapid feature development. This guide identifies **significant conceptual overlap** in UI components, data fetching patterns, and business logic that can be consolidated to reduce maintenance burden and improve code quality.

### Current State Metrics

- **124 useState calls** across 28 components
- **Only 7 uses of React Query** (useQuery/useMutation) - installed but underutilized
- **4 files** with identical 20+ line error handling logic
- **29+ files** with repetitive inline Tailwind styling
- **939 lines** for a single table component (PapersView)
- **460 lines** for NewPostForm with manual form state management
- **Zero global state management** - everything is prop drilling

---

## Critical Issues

### 🔴 1. Multiple Institution View Implementations

**Problem:** Three different institution view components with overlapping functionality:

- `InstitutionsView.tsx` - Public view with grid layout, filtering, sorting
- `UnifiedInstitutionsView.tsx` - Enhanced view with join functionality, admin controls
- `AdminInstitutionsView.tsx` - Admin-only view for managing institutions
- `ManagedInstitutionsView.tsx` - **Legacy wrapper (23 lines)** that redirects to UnifiedInstitutionsView

**Tech Debt:**

- Similar filtering/sorting logic duplicated across all three
- Inconsistent user experience between views
- ManagedInstitutionsView is pure redundancy

**Recommendation:** Consolidate into a single `InstitutionsView` with role-based rendering:

```typescript
<InstitutionsView
  userRole={currentUserRole}
  adminMode={isAdmin}
/>
```

**Files to Modify:**

- `src/components/InstitutionsView.tsx`
- `src/components/UnifiedInstitutionsView.tsx`
- `src/components/AdminInstitutionsView.tsx`
- `src/components/ManagedInstitutionsView.tsx` (DELETE)

**Estimated Savings:** ~600 lines

---

### 🔴 2. Duplicate Table/Grid Display Patterns

**Problem:** Multiple implementations of similar list/grid display logic:

#### Components with Duplication:

1. **PapersView** (939 lines)
   - Complex table with resizable columns
   - Mobile card view fallback
   - Custom drag-to-resize implementation
   - Filtering, sorting, star/queue interactions

2. **AuthorsView** (357 lines)
   - Grid layout with cards
   - Similar filtering/sorting patterns
   - Nearly identical search input UI

3. **InstitutionsView** (364 lines)
   - Grid layout with cards
   - Duplicate filtering/sorting code
   - Same search input pattern

**Common Patterns Duplicated:**

```typescript
// All three have nearly identical implementations:
- useState for searchQuery, sortBy, sortDirection
- useMemo for filtered/sorted data
- useRef for filter input
- Identical search input JSX
- Same sort button UI patterns
- formatDate utility (duplicated in 3 files)
```

**Recommendation:** Create reusable components:

- `<FilterableGrid>` with pluggable card renderers
- `<SearchInput>` with consistent styling
- `<SortControls>` for unified sort UI
- `useFilterSort()` custom hook for shared logic

**Estimated Savings:** ~800 lines

---

### 🔴 3. Overlay/Modal Implementations

**Problem:** Multiple overlay patterns with subtle differences:

- `OverlayStack` - Sophisticated stacking system with navigation
- `PaperOverlay` - Standalone overlay (not using stack)
- `NavigablePaperOverlay` - Stack-compatible version
- `NavigableAuthorOverlay` - Stack-compatible author view
- `InstitutionOverlay` - Yet another overlay pattern
- `UserCard` - Modal-like component with its own backdrop
- `DeleteConfirmationModal` - Simple confirmation modal
- `InstitutionManagementModal` - Complex management UI

**Issues:**

- Inconsistent z-index management
- Different backdrop implementations
- Duplicate loading skeletons
- Multiple ESC key handlers

**Recommendation:** Standardize on OverlayStack for all overlays, create base components:

- `<BaseOverlay>` with consistent backdrop/close logic
- `<OverlayLoadingSkeleton>` (already exists but not consistently used)
- Unified keyboard navigation handling

**Estimated Savings:** ~400 lines

---

### 🟡 4. Form Component Duplication

**Problem:** Similar form patterns across multiple files.

**Duplicated Components:**

- `InstitutionCreateForm` (243 lines)
- `GroupCreateForm` (338 lines)
- `NewPostForm` (460 lines)

**Duplicated Logic:**

- Form state management
- Error handling patterns (20+ lines identical in each)
- Validation logic
- Submit handlers using `next-safe-action`
- Loading states

**Example of Duplication:**

```typescript
// Repeated in 4+ files:
const [formData, setFormData] = useState({...})
const [error, setError] = useState<string | null>(null)

// 20+ lines of identical error parsing in each file:
const serverError = error.error?.serverError;
const validationErrors = error.error?.validationErrors;
const errorMessage =
  (typeof serverError === "string" ? serverError : null) ||
  (typeof serverError === "object" && serverError !== null && "message" in serverError
    ? (serverError as any).message : null) ||
  // ... 15 more lines
```

**Recommendation:** Adopt React Hook Form (see Library Recommendations)

**Estimated Savings:** ~1,000 lines

---

### 🟡 5. Data Fetching Redundancy

**Problem:** Multiple data loading functions with similar patterns.

**Backend Data Loaders:**

- `loadPapers()` - papers.ts
- `loadPapersWithUserContext()` - papers.ts
- `loadAuthors()` - authors-server.ts
- `loadAuthor()` - authors-server.ts
- `loadInstitutions()` - institutions-server.ts
- `loadFeedPosts()` - feed.ts
- `loadPost()` - post.ts

**Common Patterns:**

```typescript
// Every loader does:
1. Prisma query with includes
2. Transform to DTO
3. Map relations
4. Error handling with try/catch
```

**Recommendation:**

- Expand React Query usage (already installed!)
- Create generic data loading utilities
- Standardize DTO transformation
- Consider using tRPC or similar for type-safe API layer

**Estimated Savings:** ~400 lines

---

### 🟡 6. Server Action Patterns

**Problem:** 30+ server actions with repetitive boilerplate.

**Common Structure:**

```typescript
"use server"
import { actionClient } from "@/lib/actionClient"
const inputSchema = zfd.formData({ ... })
export const someAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput }) => {
    const session = await getSessionOrRedirect()
    // ... business logic
    revalidatePath("/")
    return { success: true }
  })
```

**Issues:**

- Session handling duplicated in every action
- Consistent revalidatePath patterns not extracted
- Similar error handling repeated
- Success/error response patterns vary

**Action Files:**

- `src/posts/actions/` (8 actions)
- `src/institutions/actions/` (7 actions)
- `src/groups/actions/` (4 actions)

**Recommendation:**

- Create action helper factories:
  ```typescript
  createAuthenticatedAction(schema, handler);
  createAdminAction(schema, handler);
  ```
- Standardize response types

**Estimated Savings:** ~300 lines

---

### 🟢 7. Utility Function Duplication

**Problem:** Similar utilities scattered across files.

**Duplicated Functions:**

- `formatDate()` - duplicated in at least 3 components
- `getInstitutionInitials()` - duplicated in 2 components
- `getUserInitials()` - similar pattern in MeView
- URL validation - duplicated patterns

**Recommendation:** Create `src/lib/utils/` directory:

- `date.ts` - formatDate, formatRelativeDate, etc.
- `text.ts` - getInitials, truncate, etc.
- `validation.ts` - isValidUrl, etc.

**Estimated Savings:** ~100 lines

---

### 🟢 8. Type Definition Overlap

**Problem:** Similar type definitions in multiple files.

**Duplicated Types:**

- `Paper`, `PaperWithUserContext`, `PaperDTO` - slightly different
- `Author`, `AuthorDTO` - transformation overhead
- `Institution`, `InstitutionDTO`, `UnifiedInstitutionDTO` - **3 versions!**
- `User`, `UserInteraction` - scattered definitions

**Recommendation:**

- Consolidate into `src/types/` directory
- Use Prisma-generated types as source of truth
- Create transformation utilities instead of new types

**Estimated Savings:** ~200 lines

---

## Library Recommendations

### Priority 1: High-Impact Libraries

#### 1. React Hook Form (🔥 Top Priority)

**Impact:** ~800-1,000 line reduction
**Effort:** 1-2 weeks
**Status:** Not installed

**Why:**

- Already using Zod for validation
- Works seamlessly with `next-safe-action`
- Eliminates all manual form state management
- Built-in error handling and validation

**Installation:**

```bash
pnpm add react-hook-form @hookform/resolvers
```

**Specific Wins:**

- Eliminate 80+ lines in NewPostForm
- Eliminate 60+ lines in InstitutionCreateForm
- Eliminate 80+ lines in GroupCreateForm
- Eliminate 100+ lines in InstitutionManagementModal

**Example Usage:**

```typescript
const form = useForm({
  resolver: zodResolver(schema),
  defaultValues: {...}
})

<Form {...form}>
  <FormField name="name" />
  <FormField name="description" />
</Form>
```

---

#### 2. TanStack Table v8

**Impact:** ~600-700 line reduction
**Effort:** 1 week
**Status:** Not installed

**Why:**

- PapersView.tsx is 939 lines with custom table logic
- Column resizing, sorting, filtering all built-in
- Fully typed with TypeScript
- Works well with React Query

**Installation:**

```bash
pnpm add @tanstack/react-table
```

**Specific Wins:**

- Reduce PapersView from 939 lines to ~200 lines
- Eliminate custom column resizing logic (100+ lines)
- Eliminate custom sorting logic (80+ lines)
- Eliminate custom filtering logic (60+ lines)

**Example Usage:**

```typescript
const table = useReactTable({
  data: papers,
  columns,
  getCoreRowModel: getCoreRowModel(),
  getSortedRowModel: getSortedRowModel(),
  getFilteredRowModel: getFilteredRowModel(),
});
```

---

#### 3. shadcn/ui

**Impact:** ~500-600 line reduction
**Effort:** 2-3 weeks
**Status:** Not installed

**Why:**

- 29+ components with complex inline Tailwind
- Inconsistent modal/dialog implementations
- Manual accessibility (aria labels, focus management)
- Built on Radix UI (excellent a11y)

**Installation:**

```bash
pnpm add @radix-ui/react-dialog @radix-ui/react-popover
npx shadcn-ui@latest init
npx shadcn-ui@latest add dialog input button select
```

**Key Components to Adopt:**

- `Dialog` - replace all 8 modal implementations
- `Input` / `Textarea` / `Select` - consistent form inputs
- `Button` - replace custom Button.tsx
- `Popover` - for UserCard and similar
- `Command` - for mention autocomplete
- `Tabs` - cleaner organization in settings/admin views

**Specific Wins:**

- InstitutionManagementModal: 441 → 180 lines
- GroupManagementModal: 362 → 150 lines
- Standardize all modal implementations

**Example Usage:**

```typescript
<Dialog>
  <DialogTrigger>Open</DialogTrigger>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Title</DialogTitle>
    </DialogHeader>
    <Input placeholder="Name" />
    <Button variant="default">Submit</Button>
  </DialogContent>
</Dialog>
```

---

#### 4. Better React Query Usage

**Impact:** ~300-400 line reduction + better UX
**Effort:** 1-2 weeks
**Status:** ✅ Installed but underutilized (only 7 uses)

**Why:**

- Already installed and configured!
- All data loaded on server, passed as props through 5+ levels
- Manual loading states everywhere
- No optimistic updates (except star widget)
- Stale data after mutations

**Current Flow:**

```typescript
// Page.tsx - server component
const { papers, users, interactions } = await loadPapersWithUserContext()

// Props through 3 levels
<PapersView papers={papers} users={users} interactions={interactions} />
  <PaperCard paper={paper} /> // needs all 3 props
    <StarWidget paper={paper} users={users} /> // needs all props
```

**Better Flow:**

```typescript
// In any component:
const { data: papers } = useQuery({
  queryKey: ["papers"],
  queryFn: () => fetch("/api/papers").then((r) => r.json()),
});

// No prop drilling needed!
```

**Benefits:**

- Eliminate prop drilling (remove 100s of prop declarations)
- Automatic loading/error states
- Optimistic updates for all mutations
- Background refetching for stale data
- Request deduplication

---

### Priority 2: Medium-Impact Libraries

#### 5. Zustand (Lightweight State Management)

**Impact:** ~200-300 line reduction
**Effort:** 0.5-1 week
**Status:** Not installed

**Why:**

- Zero global state management currently
- OverlayStackContext could be simpler
- Filter/sort preferences duplicated across components

**Installation:**

```bash
pnpm add zustand
```

**Use Cases:**

```typescript
// Store for overlay/modal state
const useOverlayStore = create((set) => ({
  overlays: [],
  openOverlay: (overlay) =>
    set((state) => ({
      overlays: [...state.overlays, overlay],
    })),
  closeOverlay: () =>
    set((state) => ({
      overlays: state.overlays.slice(0, -1),
    })),
}));

// Store for filter/sort preferences (persisted)
const useFilterStore = create(
  persist(
    (set) => ({
      searchQuery: "",
      sortBy: "name",
      sortDirection: "asc",
    }),
    { name: "filter-preferences" }
  )
);
```

**Specific Wins:**

- Replace OverlayStackContext (88 lines) → 30 lines
- Centralize filter state from 6+ components
- Persist user preferences

---

#### 6. TanStack Virtual

**Impact:** ~100 line reduction + performance
**Effort:** 0.5 week
**Status:** Not installed

**Why:**

- Large lists (feed with 100+ posts, papers with 500+ entries)
- Custom InfiniteScroll implementation can be replaced

**Installation:**

```bash
pnpm add @tanstack/react-virtual
```

**Example Usage:**

```typescript
const virtualizer = useVirtualizer({
  count: papers.length,
  getScrollElement: () => parentRef.current,
  estimateSize: () => 120,
});

// Renders only visible items
```

---

#### 7. Sonner (Toast Notifications)

**Impact:** ~50 line reduction + better UX
**Effort:** 0.5 week
**Status:** Not installed

**Why:**

- Manual error/success messages in every form
- Inconsistent notification patterns

**Installation:**

```bash
pnpm add sonner
```

**Example Usage:**

```typescript
import { toast } from "sonner";

toast.success("Institution created!");
toast.error("Failed to join institution");

// Instead of managing error state in every form
```

---

#### 8. CVA (Class Variance Authority)

**Impact:** ~200 line reduction
**Effort:** 1 week
**Status:** Not installed

**Why:**

- Standardize Tailwind patterns
- Type-safe variant system
- Works great with shadcn/ui

**Installation:**

```bash
pnpm add class-variance-authority
```

**Example Usage:**

```typescript
const buttonVariants = cva(
  "rounded-lg font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "bg-gray-100 hover:bg-gray-200",
        primary: "bg-blue-600 hover:bg-blue-700 text-white",
        danger: "bg-red-600 hover:bg-red-700 text-white",
      },
      size: {
        sm: "px-3 py-1.5 text-sm",
        md: "px-4 py-2",
        lg: "px-6 py-3 text-lg",
      },
    },
  }
)

<button className={buttonVariants({ variant: "primary", size: "md" })}>
```

---

## Line Reduction Strategies

### Strategy 1: Extract Repeated Patterns → Custom Hooks

**Estimated Reduction:** ~500 lines

#### Filter/Sort Pattern

**Duplicated in:** AuthorsView, InstitutionsView, UnifiedInstitutionsView, PapersView, GroupsView, AdminInstitutionsView

**Current (50 lines per component × 6 = 300 lines):**

```typescript
const [searchQuery, setSearchQuery] = useState("");
const [sortBy, setSortBy] = useState("name");
const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

const filteredAndSorted = useMemo(() => {
  let filtered = items.filter((item) =>
    item.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return filtered.sort((a, b) => {
    // 20+ lines of sort logic
  });
}, [items, searchQuery, sortBy, sortDirection]);
```

**Extract to Hook (50 lines once):**

```typescript
// src/lib/hooks/useFilterSort.ts
function useFilterSort<T>(
  items: T[],
  searchFields: (keyof T)[],
  sortFields: Record<string, (item: T) => any>
) {
  // Generic implementation
}

// Usage:
const { filtered, searchQuery, setSearchQuery, sortBy, setSortBy } =
  useFilterSort(papers, ["title", "tags"], {
    title: (p) => p.title,
    date: (p) => p.createdAt,
  });
```

**Files to Create:**

- `src/lib/hooks/useFilterSort.ts`
- `src/lib/hooks/useErrorHandling.ts` (save 80 lines)
- `src/lib/hooks/useOptimisticMutation.ts` (save 100 lines)
- `src/lib/hooks/useModal.ts` (save 50 lines)

---

### Strategy 2: Component Composition > Duplication

**Estimated Reduction:** ~600 lines

#### Card Pattern

**Duplicated in:** ~10 card implementations (InstitutionCard, AuthorCard, etc.)

**Current (80-120 lines per card × 10 = ~1000 lines):**

```typescript
<div className="rounded-lg border border-gray-200 bg-white p-6">
  <div className="flex items-start gap-4">
    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-600">
      {/* Avatar */}
    </div>
    <div className="flex-1">
      <h3>{/* Title */}</h3>
      <p>{/* Subtitle */}</p>
    </div>
  </div>
  {/* Stats */}
  {/* Tags */}
  {/* Actions */}
</div>
```

**Compose Instead (~400 lines total):**

```typescript
// src/components/ui/Card.tsx
<Card>
  <Card.Header>
    <Card.Avatar>{initials}</Card.Avatar>
    <Card.Title>{name}</Card.Title>
    <Card.Subtitle>{subtitle}</Card.Subtitle>
  </Card.Header>
  <Card.Stats>
    <Stat label="Papers" value={count} />
  </Card.Stats>
  <Card.Tags tags={tags} />
  <Card.Actions>{/* ... */}</Card.Actions>
</Card>
```

**Files to Create:**

- `src/components/ui/Card.tsx`
- `src/components/ui/Stat.tsx`

---

### Strategy 3: Code Generation for Repetitive Patterns

**Estimated Reduction:** ~300 lines

**Candidate for Generation:**

- CRUD actions (createX, updateX, deleteX patterns identical)
- Data loaders (loadX, loadXById patterns identical)
- API routes (most are thin wrappers)

**Example:**

```typescript
// scripts/generate-crud.ts
generateCRUD("institution", {
  fields: ["name", "domain", "description"],
  hasOwner: true,
  requiresAuth: true,
});

// Generates:
// - src/institutions/actions/createInstitution.ts
// - src/institutions/actions/updateInstitution.ts
// - src/institutions/actions/deleteInstitution.ts
// - src/api/institutions/route.ts
```

**Files to Create:**

- `scripts/generate-crud.ts`

---

### Strategy 4: Replace Large Components with Libraries

**Estimated Reduction:** ~1,200 lines

| Component                  | Current Lines | With Library          | Savings |
| -------------------------- | ------------- | --------------------- | ------- |
| PapersView (table)         | 939           | 200 (TanStack Table)  | **739** |
| NewPostForm                | 460           | 150 (React Hook Form) | **310** |
| InstitutionManagementModal | 441           | 180 (shadcn + RHF)    | **261** |
| GroupManagementModal       | 362           | 150 (shadcn + RHF)    | **212** |
| InstitutionCreateForm      | 243           | 80 (React Hook Form)  | **163** |
| GroupCreateForm            | 338           | 100 (React Hook Form) | **238** |

**Total: 1,923 line reduction**

---

### Strategy 5: Eliminate Dead/Redundant Code

**Estimated Reduction:** ~200 lines

**Found:**

- `ManagedInstitutionsView.tsx` - **23 line wrapper (DELETE)**
- Unused utilities in NewPostForm (complex quote logic)
- Duplicate `formatDate` in 3+ files (consolidate)
- Custom `usePaginator` when React Query has built-in infinite queries

**Files to Delete:**

- `src/components/ManagedInstitutionsView.tsx`

**Files to Consolidate:**

- Create `src/lib/utils/date.ts`
- Create `src/lib/utils/text.ts`
- Create `src/lib/utils/validation.ts`

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Impact:** ~800 lines saved

#### Tasks:

1. ✅ Install and configure shadcn/ui

   ```bash
   npx shadcn-ui@latest init
   npx shadcn-ui@latest add dialog input button select textarea
   ```

2. ✅ Extract error handling utilities
   - Create `src/lib/utils/errors.ts`
   - Extract 20+ line error parsing logic
   - Update 4 files to use utility

3. ✅ Create shared hooks
   - `src/lib/hooks/useFilterSort.ts`
   - `src/lib/hooks/useErrorHandling.ts`

4. ✅ Delete dead code
   - Delete `src/components/ManagedInstitutionsView.tsx`
   - Update imports in pages

5. ✅ Consolidate utility functions
   - Create `src/lib/utils/date.ts`
   - Create `src/lib/utils/text.ts`
   - Create `src/lib/utils/validation.ts`
   - Update all files using these utilities

#### Files to Modify:

- `src/components/InstitutionCreateForm.tsx`
- `src/components/GroupCreateForm.tsx`
- `src/components/InstitutionManagementModal.tsx`
- `src/components/GroupManagementModal.tsx`
- `src/components/ManagedInstitutionsView.tsx` (DELETE)

---

### Phase 2: Forms (Week 3-4)

**Impact:** ~1,300 lines saved

#### Tasks:

1. ✅ Install React Hook Form

   ```bash
   pnpm add react-hook-form @hookform/resolvers
   ```

2. ✅ Create form components with shadcn/ui
   - `src/components/ui/Form.tsx`
   - `src/components/ui/FormField.tsx`

3. ✅ Migrate NewPostForm (biggest win - 310 lines saved)
   - Replace manual state management
   - Use RHF with Zod validation
   - Use shadcn/ui form components

4. ✅ Migrate InstitutionCreateForm (163 lines saved)
   - Convert to RHF
   - Use shadcn/ui Dialog

5. ✅ Migrate GroupCreateForm (238 lines saved)
   - Convert to RHF
   - Use shadcn/ui Dialog

6. ✅ Migrate management modals (473 lines saved)
   - InstitutionManagementModal
   - GroupManagementModal
   - Use shadcn/ui Dialog + RHF

#### Files to Modify:

- `src/app/NewPostForm.tsx`
- `src/components/InstitutionCreateForm.tsx`
- `src/components/GroupCreateForm.tsx`
- `src/components/InstitutionManagementModal.tsx`
- `src/components/GroupManagementModal.tsx`

---

### Phase 3: Data Layer (Week 4-5)

**Impact:** ~700 lines saved + better UX

#### Tasks:

1. ✅ Create API endpoints for client-side fetching
   - Convert server loaders to API routes where needed
   - Ensure consistent response formats

2. ✅ Expand React Query usage
   - Convert PapersView to use useQuery
   - Convert AuthorsView to use useQuery
   - Convert InstitutionsView to use useQuery
   - Remove prop drilling patterns

3. ✅ Install and configure Zustand

   ```bash
   pnpm add zustand
   ```

4. ✅ Create Zustand stores
   - `src/lib/stores/overlayStore.ts`
   - `src/lib/stores/filterStore.ts`
   - Replace OverlayStackContext

5. ✅ Implement optimistic updates
   - Use React Query mutations with optimistic updates
   - Remove manual state management

#### Files to Modify:

- `src/components/PapersView.tsx`
- `src/components/AuthorsView.tsx`
- `src/components/InstitutionsView.tsx`
- `src/components/OverlayStack.tsx`
- `src/app/papers/page.tsx`
- `src/app/authors/page.tsx`
- `src/app/institutions/page.tsx`

---

### Phase 4: Complex Components (Week 6-7)

**Impact:** ~1,500 lines saved

#### Tasks:

1. ✅ Install TanStack Table

   ```bash
   pnpm add @tanstack/react-table
   ```

2. ✅ Migrate PapersView to TanStack Table (739 lines saved)
   - Replace custom table logic
   - Use built-in column resizing
   - Use built-in sorting/filtering

3. ✅ Create Card composition components
   - `src/components/ui/Card.tsx`
   - Update all card usages

4. ✅ Consolidate institution views
   - Merge InstitutionsView, UnifiedInstitutionsView, AdminInstitutionsView
   - Use role-based rendering

5. ✅ Install and add TanStack Virtual
   ```bash
   pnpm add @tanstack/react-virtual
   ```

   - Add to feed view
   - Add to papers view
   - Replace InfiniteScroll

#### Files to Modify:

- `src/components/PapersView.tsx`
- `src/components/InstitutionsView.tsx`
- `src/components/UnifiedInstitutionsView.tsx`
- `src/components/AdminInstitutionsView.tsx`
- `src/app/FeedPostsPage.tsx`

---

### Phase 5: Polish (Week 8)

**Impact:** ~300 lines saved + better DX

#### Tasks:

1. ✅ Install Sonner

   ```bash
   pnpm add sonner
   ```

   - Add to layout
   - Replace manual error messages with toasts

2. ✅ Install CVA

   ```bash
   pnpm add class-variance-authority
   ```

   - Create variant systems for common components

3. ✅ Create code generation scripts
   - `scripts/generate-crud.ts`
   - Document usage

4. ✅ Update documentation
   - Update README with new patterns
   - Document new hooks and utilities
   - Add examples

#### Files to Create/Modify:

- `src/app/layout.tsx` (add Sonner)
- `src/lib/variants/button.ts`
- `src/lib/variants/card.ts`
- `scripts/generate-crud.ts`
- `README.md`

---

## Quick Wins

If you want incremental improvements, start here (< 1 week each):

### 1. Delete ManagedInstitutionsView (5 minutes)

**Savings:** 23 lines

```bash
rm src/components/ManagedInstitutionsView.tsx
# Update imports to use UnifiedInstitutionsView
```

**Files to Update:**

- Any page importing ManagedInstitutionsView

---

### 2. Extract Error Handling (2-3 hours)

**Savings:** ~80 lines

**Create `src/lib/utils/errors.ts`:**

```typescript
export function extractActionError(error: any): string {
  const serverError = error.error?.serverError;
  const validationErrors = error.error?.validationErrors;

  return (
    (typeof serverError === "string" ? serverError : null) ||
    (typeof serverError === "object" &&
    serverError !== null &&
    "message" in serverError
      ? (serverError as any).message
      : null) ||
    (Array.isArray(validationErrors) &&
    validationErrors.length > 0 &&
    typeof validationErrors[0] === "object" &&
    validationErrors[0] !== null &&
    "message" in validationErrors[0]
      ? (validationErrors[0] as any).message
      : null) ||
    "An error occurred. Please try again."
  );
}
```

**Files to Update:**

- `src/components/InstitutionCreateForm.tsx`
- `src/components/GroupCreateForm.tsx`
- `src/components/InstitutionManagementModal.tsx`
- `src/components/GroupManagementModal.tsx`

---

### 3. Consolidate Format Date (1 hour)

**Savings:** ~30 lines

**Create `src/lib/utils/date.ts`:**

```typescript
export function formatDate(date: Date | string | null): string {
  if (!date) return "Unknown";
  const dateObj = typeof date === "string" ? new Date(date) : date;
  if (isNaN(dateObj.getTime())) return "Unknown";

  const now = new Date();
  const diffInDays = Math.floor(
    (now.getTime() - dateObj.getTime()) / (1000 * 60 * 60 * 24)
  );

  if (diffInDays === 0) return "Today";
  if (diffInDays === 1) return "Yesterday";
  if (diffInDays < 7) return `${diffInDays} days ago`;
  if (diffInDays < 30) return `${Math.floor(diffInDays / 7)} weeks ago`;
  return `${Math.floor(diffInDays / 30)} months ago`;
}
```

**Files to Update:**

- `src/components/UnifiedInstitutionsView.tsx`
- `src/components/AdminInstitutionsView.tsx`
- Any other files with formatDate

---

### 4. Add shadcn/ui Dialog (1 day)

**Savings:** ~400 lines

**Installation:**

```bash
npx shadcn-ui@latest init
npx shadcn-ui@latest add dialog
```

**Standardize all 8 modal implementations:**

- InstitutionManagementModal
- GroupManagementModal
- InstitutionCreateForm (modal part)
- GroupCreateForm (modal part)
- DeleteConfirmationModal
- And others

---

### 5. Migrate NewPostForm to React Hook Form (2-3 days)

**Savings:** ~310 lines

**Installation:**

```bash
pnpm add react-hook-form @hookform/resolvers
```

**Biggest single-file win!**

---

## Detailed Analysis

### Backend Analysis

**API Routes Structure:**

```
/api/
  ├── admin/ (2 routes)
  ├── analyze-pdf/
  ├── authors/ (2 routes)
  ├── chat/
  ├── discord/ (3 routes)
  ├── institutions/ (1 route)
  ├── mentions/
  ├── notifications/ (3 routes)
  ├── papers/ (2 routes)
  └── posts/ (2 routes)
```

**Observations:**

- ✅ **Good:** Clean organization by entity
- ⚠️ **Concern:** Some routes are very thin wrappers around data functions
- 💡 **Opportunity:** Many routes just do `loadX() → JSON`, could be generated

---

### Metrics Summary

**Total Estimated Reduction:**

| Strategy                 | Line Reduction  | Effort (weeks) |
| ------------------------ | --------------- | -------------- |
| React Hook Form adoption | **1,000 lines** | 1-2            |
| TanStack Table           | **700 lines**   | 1              |
| shadcn/ui components     | **600 lines**   | 2-3            |
| React Query expansion    | **400 lines**   | 1-2            |
| Custom hooks extraction  | **500 lines**   | 1              |
| Component composition    | **600 lines**   | 1-2            |
| Zustand for state        | **300 lines**   | 0.5            |
| Code generation          | **300 lines**   | 1              |
| Dead code removal        | **200 lines**   | 0.5            |

**Total: ~4,600 lines (40-45% of UI layer)**

---

### Cost-Benefit Analysis

**Costs:**

- 6-8 weeks development time
- Learning curve for new libraries
- Temporary disruption to feature velocity
- Testing overhead

**Benefits:**

- **45% less code to maintain**
- Faster feature development (shadcn/ui, RHF)
- Better UX (React Query caching, optimistic updates)
- Fewer bugs (library-tested patterns)
- Better performance (virtualization, proper memoization)
- Easier onboarding (standard patterns)

**Break-even:** ~3-4 months after completion

---

## Next Steps

1. **Review this document** with your team
2. **Choose a starting phase** (recommend Phase 1: Foundation)
3. **Create feature branch** for refactoring
4. **Write tests** for existing behavior before refactoring
5. **Incremental commits** with frequent reviews
6. **Track progress** by checking off tasks in this document

---

## Notes

- This document will be updated as refactoring progresses
- Mark completed tasks with ✅
- Add any new findings or issues discovered during refactoring
- Keep track of actual time vs estimated time for future planning
