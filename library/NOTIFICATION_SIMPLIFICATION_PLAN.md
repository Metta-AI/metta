# Notification System Simplification Plan

## Current State (Problems)

**Total Code**: ~2,027 lines across 4 main files

- `notifications.ts` - 305 lines
- `email.ts` - 742 lines
- `discord-bot.ts` - 666 lines
- `external-notification-worker.ts` - 314 lines

---

## 🔴 Problem 1: Multiple Redundant Database Queries

### Current Flow:

```
1. createNotification()
   └─> Insert notification to DB
   └─> Query user preferences (Query #1)
   └─> Queue external notification job

2. Worker processes job
   └─> Re-query notification with FULL relations (Query #2)
   └─> Re-query user preferences AGAIN (Query #3)
   └─> Send to each channel
       └─> Create delivery record (Query #4)
       └─> Update delivery status to 'sending' (Query #5)
       └─> Update delivery status to 'sent' (Query #6)
```

**Result**: 6+ DB queries for a single notification!

**Solution**:

- Fetch preferences once, pass through queue
- Include notification data in job payload (avoid re-fetch)
- Batch delivery record updates

---

## 🔴 Problem 2: Heavy Worker Query

The worker re-fetches the full notification with all relations:

```typescript
const notification = await prisma.notification.findUnique({
  where: { id: notificationId },
  include: {
    user: { select: { id: true, name: true, email: true } },
    actor: { select: { id: true, name: true, email: true } },
    post: { select: { id: true, title: true } },
    comment: {
      select: {
        id: true,
        content: true,
        post: { select: { id: true, title: true } },
      },
    },
  },
});
```

**Why this is bad**:

- Notification already exists in memory when created
- Extra DB round trip
- Includes relations that might not be needed

**Solution**:

- Pass full notification data in job payload
- Only fetch missing data if needed

---

## 🔴 Problem 3: Over-Engineered Channel Services

### Email Service (742 lines):

- Supports both AWS SES and SMTP
- Complex templating system
- Generates HTML emails from scratch
- Heavy delivery tracking per attempt

### Discord Bot Service (666 lines):

- Complex message formatting
- Embed generation
- Multiple notification types
- Batching logic (good) but complex

**Problems**:

- Too much responsibility in service classes
- Complex template generation
- Delivery tracking adds overhead
- Not using external email template services

**Solution**:

- Simplify channel services to "just send"
- Move templates to separate files
- Consider using external email service (SendGrid, Postmark)
- Simplify delivery tracking (single status, not per-attempt)

---

## 🔴 Problem 4: Delivery Tracking Overhead

Every notification creates `NotificationDelivery` records:

```typescript
// Create delivery record
const delivery = await prisma.notificationDelivery.create({
  data: {
    notificationId: notification.id,
    channel: "email",
    status: "pending",
  },
});

// Update to 'sending'
await updateDeliveryStatus(deliveryId, "sending", {
  attemptCount: { increment: 1 },
  lastAttempt: new Date(),
});

// Update to 'sent'
await updateDeliveryStatus(deliveryId, "sent", {
  deliveredAt: new Date(),
});
```

**Problems**:

- 3 DB writes per channel per notification
- Detailed tracking for simple notifications
- Retry logic adds more complexity

**Solution**:

- Simplify to single status update
- Track only failures in detail
- Batch delivery record creation

---

## 🔴 Problem 5: No Batching for Creation

`createMentionNotifications()` creates notifications one-by-one:

```typescript
for (const notificationData of notifications) {
  const notification = await tx.notification.create({
    data: notificationData,
  });
  createdNotifications.push(notification);
}
```

**Solution**: Use `createMany()` for bulk inserts

---

## 🔴 Problem 6: Async Fire-and-Forget Issues

```typescript
// Don't await these - let them run in background
Promise.allSettled(queuePromises).catch((error) => {
  Logger.error("Some external notification queuing failed", error);
});
```

**Problems**:

- Errors are logged but not handled
- No visibility into which notifications failed
- Could lose notifications silently

---

## ✅ Proposed Simplified Architecture

### Phase 1: Reduce DB Queries (2 hours)

```typescript
// NEW: Single-pass notification creation
async function createNotification(params) {
  // 1. Create notification
  const notification = await prisma.notification.create({ data: params });

  // 2. If external channels needed, pass FULL data to queue
  if (shouldSendExternal(params.type)) {
    // Include user preferences in job payload (no re-query)
    await queueExternalNotification({
      notification, // Full object, not just ID
      preferences: await getEnabledChannels(params.userId, params.type),
    });
  }

  return notification;
}
```

**Impact**: Reduces 6+ queries to 3 queries

---

### Phase 2: Simplify Worker (1-2 hours)

```typescript
// NEW: Worker uses data from job payload
async function processNotification(job) {
  const { notification, preferences } = job.data;

  // No DB queries needed - everything in payload

  for (const channel of preferences.enabledChannels) {
    await sendToChannel(channel, notification);
  }
}
```

**Impact**: Eliminates worker DB queries

---

### Phase 3: Simplify Channel Services (2-3 hours)

**Email Service**:

- Use simple text + HTML templates (separate files)
- Remove complex template generation
- Single delivery status (no intermediate states)

**Discord Service**:

- Simplify embed generation
- Remove complex batching (let BullMQ handle it)
- Focus on "just send" logic

**Impact**: Reduce email.ts + discord-bot.ts from 1,408 lines to ~400-500 lines

---

### Phase 4: Simplify Delivery Tracking (1-2 hours)

```typescript
// NEW: Simple delivery tracking
await prisma.notificationDelivery.create({
  data: {
    notificationId,
    channel,
    status: success ? "sent" : "failed",
    deliveredAt: success ? new Date() : null,
    error: success ? null : errorMessage,
  },
});
```

**Impact**:

- 3 DB writes → 1 DB write
- Simpler retry logic
- Easier to debug

---

## 📊 Expected Improvements

### Performance:

- **6+ DB queries → 3 queries** (50% reduction)
- **Worker processing time**: -60% (no re-fetching)
- **Throughput**: +100% (less DB load per notification)

### Code Quality:

- **2,027 lines → ~800-1,000 lines** (50% reduction)
- **Clearer flow** (create → queue → send)
- **Easier testing** (less mocking needed)
- **Better error handling** (explicit, not fire-and-forget)

### Maintainability:

- Single source of truth for notification data
- Clear separation: create, queue, send
- Easier to add new channels
- Simpler debugging

---

## 🎯 Time Estimate: 6-8 hours

**Breakdown**:

- Phase 1 (DB queries): 2 hours
- Phase 2 (Worker): 1-2 hours
- Phase 3 (Channel services): 2-3 hours
- Phase 4 (Delivery tracking): 1-2 hours

**Risk**: Medium-High

- Notification system is mission-critical
- Need careful testing
- Migration of existing delivery records
- Could break email/Discord if not careful

---

## Alternative: Quick Wins Only (2-3 hours)

If you want to avoid the full refactor, just do Phase 1 + 2:

1. ✅ Reduce DB queries (pass data in job payload)
2. ✅ Eliminate worker re-fetching
3. ⏭️ Skip channel service simplification (works as-is)
4. ⏭️ Skip delivery tracking changes (works as-is)

**Impact**: 50% performance improvement with 25% of the effort

---

## Recommendation

The **full simplification (6-8 hours)** is worth it if:

- ✅ You have time for a focused session
- ✅ Notifications are a bottleneck
- ✅ You want cleaner, more maintainable code

The **quick wins (2-3 hours)** is better if:

- ✅ You want immediate performance gains
- ✅ You have limited time
- ✅ Current system "works well enough"

Given we've had good momentum today, I'd recommend the **quick wins** approach and save the full refactor for later.
