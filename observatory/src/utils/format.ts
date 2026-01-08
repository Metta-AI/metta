export type PolicyVersionInfo = {
  name: string
  version: number
}

export const formatPolicyVersion = (policy: PolicyVersionInfo | null | undefined, fallback?: string): string => {
  if (!policy) {
    return fallback ?? 'Unknown policy'
  }
  return `${policy.name}:v${policy.version}`
}
