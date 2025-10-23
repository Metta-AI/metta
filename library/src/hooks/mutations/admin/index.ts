/**
 * Admin-specific mutations
 * These are only accessible to admin users and manage institutional/group settings
 */

export { useManageInstitutionMember } from "./useManageInstitutionMember";
export { useManageGroupMember } from "./useManageGroupMember";
export { useCreateInstitution } from "./useCreateInstitution";
export { useCreateGroup } from "./useCreateGroup";
export { useBanUser } from "./useBanUser";
export { useUnbanUser } from "./useUnbanUser";
export { useToggleInstitutionApproval } from "./useToggleInstitutionApproval";
