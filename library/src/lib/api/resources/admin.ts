import { fetchJson } from "@/lib/api/client";

export interface AdminInstitution {
  id: string;
  name: string;
  domain: string | null;
  memberCount: number;
  paperCount: number;
  requiresApproval: boolean;
  pendingMembersCount: number;
  owners: Array<{
    id: string;
    name: string | null;
    email: string | null;
  }>;
}

export interface AdminUser {
  id: string;
  name: string | null;
  email: string | null;
  emailVerified: Date | null;
  image: string | null;
  isBanned: boolean;
  bannedAt: Date | null;
  banReason: string | null;
  bannedBy: {
    id: string;
    name: string | null;
    email: string | null;
  } | null;
  createdAt: string;
  postCount: number;
  commentCount: number;
  institutionCount: number;
}

export interface AdminUserListResponse {
  users: AdminUser[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export async function listAdminInstitutions(): Promise<AdminInstitution[]> {
  const response = await fetchJson<{ institutions: AdminInstitution[] }>(
    "/api/admin/institutions"
  );
  return response.institutions;
}

export async function listAdminUsers(params?: {
  page?: number;
  limit?: number;
  search?: string;
  banned?: boolean;
}): Promise<AdminUserListResponse> {
  const queryParams = new URLSearchParams();
  if (params?.page) queryParams.set("page", String(params.page));
  if (params?.limit) queryParams.set("limit", String(params.limit));
  if (params?.search) queryParams.set("search", params.search);
  if (params?.banned !== undefined)
    queryParams.set("banned", String(params.banned));

  const url =
    queryParams.size > 0
      ? `/api/admin/users?${queryParams.toString()}`
      : "/api/admin/users";
  return fetchJson<AdminUserListResponse>(url);
}
