/**
 * Centralized mutations using React Query patterns
 * All mutations follow consistent patterns for:
 * - Optimistic updates
 * - Error handling
 * - Cache invalidation
 */

// Core user mutations
export { useStarMutation } from "../useStarMutation";
export { useCreatePost } from "./useCreatePost";
export { useDeletePost } from "./useDeletePost";
export { useCreateComment } from "./useCreateComment";
export { useDeleteComment } from "./useDeleteComment";
export { useJoinInstitution } from "./useJoinInstitution";
export { useJoinGroup } from "./useJoinGroup";

// Settings & notifications
export { useMarkNotificationsRead } from "./useMarkNotificationsRead";
export { useUpdateNotificationPreferences } from "./useUpdateNotificationPreferences";
export { useUnlinkDiscord } from "./useUnlinkDiscord";

// Admin mutations
export * from "./admin";
