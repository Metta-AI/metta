// src/config.ts
interface Config {
  apiBaseUrl: string
  authToken?: string
}

export const config: Config = {
  apiBaseUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  authToken: import.meta.env.VITE_AUTH_TOKEN,
}
