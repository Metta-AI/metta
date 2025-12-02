// src/config.ts
interface Config {
  apiBaseUrl: string
  authServerUrl: string
  authToken?: string
}

export const config: Config = {
  apiBaseUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  authToken: import.meta.env.VITE_AUTH_TOKEN,
  authServerUrl: import.meta.env.VITE_AUTH_SERVER_URL || 'https://softmax.com/api',
}
