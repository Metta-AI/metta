// Authentication module for Observatory
// Manages OAuth tokens in localStorage

import { config } from './config'

const TOKEN_STORAGE_KEY = 'observatory_auth_token'

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

/**
 * Get the auth server URL (where to redirect for login)
 */
export function getAuthServerUrl(): string {
  // Use the same default as CoGames CLI: https://softmax.com/api
  return import.meta.env.VITE_API_URL || 'https://softmax.com/api'
}

/**
 * Initiate the login flow by redirecting to the auth server
 */
export function initiateLogin(): void {
  const authServerUrl = getAuthServerUrl()
  const callbackUrl = `${window.location.origin}/auth/callback`

  // Build the authentication URL
  const params = new URLSearchParams({
    callback: callbackUrl,
  })

  const authUrl = `${authServerUrl}/tokens/cli?${params.toString()}`

  // Redirect to the auth server
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
