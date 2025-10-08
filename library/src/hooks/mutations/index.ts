/**
 * Centralized mutations using React Query patterns
 * All mutations follow consistent patterns for:
 * - Optimistic updates
 * - Error handling
 * - Cache invalidation
 */

// Core user mutations
export { useStarMutation } from "../useStarMutation";
export { useQueuePaper } from "./useQueuePaper";
export { useDeletePost } from "./useDeletePost";
export { useCreateComment } from "./useCreateComment";
export { useDeleteComment } from "./useDeleteComment";

// Settings & notifications
export { useMarkNotificationsRead } from "./useMarkNotificationsRead";
export { useUpdateNotificationPreferences } from "./useUpdateNotificationPreferences";
export { useUnlinkDiscord } from "./useUnlinkDiscord";

// Admin mutations
export * from "./admin";
