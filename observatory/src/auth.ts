// Authentication module for Observatory
// Manages OAuth tokens in localStorage

import { config } from './config'

const TOKEN_STORAGE_KEY = 'observatory_auth_token'

let isRedirectingToLogin = false

/**
 * Get the current auth token from localStorage
 */
export function getToken(): string | null {
  return config.authToken || localStorage.getItem(TOKEN_STORAGE_KEY)
}

/**
 * Save the auth token to localStorage
 */
export function setToken(token: string): void {
  localStorage.setItem(TOKEN_STORAGE_KEY, token)
}

/**
 * Remove the auth token from localStorage
 */
export function clearToken(): void {
  localStorage.removeItem(TOKEN_STORAGE_KEY)
}

/**
 * Check if we have a saved token
 */
export function hasToken(): boolean {
  return getToken() !== null
}

export function isRedirecting(): boolean {
  return isRedirectingToLogin
}

export function initiateLogin(): void {
  if (isRedirectingToLogin) {
    return
  }
  isRedirectingToLogin = true

  const callbackUrl = `${window.location.origin}/auth/callback`
  const params = new URLSearchParams({
    callback: callbackUrl,
  })
  const authUrl = `${config.authServerUrl}/tokens/cli?${params.toString()}`
  window.location.href = authUrl
}

/**
 * Handle logout
 */
export function logout(): void {
  clearToken()
  // Redirect to login
  initiateLogin()
}
