export interface PolicyInfo {
  id: string;
  type: "training_run" | "policy";
  name: string;
  user_id?: string;
  created_at: string;
  tags: string[];
}

export interface PolicySelectorData {
  policies: PolicyInfo[];
  selectedPolicies: string[];
}

export interface FilterState {
  searchTerm: string;
  policyTypeFilter: string[];
  tagFilter: string[];
}

export interface UIConfig {
  showTags: boolean;
  showType: boolean;
  showCreatedAt: boolean;
  maxDisplayedPolicies: number;
}