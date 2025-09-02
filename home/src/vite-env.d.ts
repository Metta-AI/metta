/// <reference types="vite/client" />

declare module '*.yaml' {
  const content: {
    links: Array<{
      name: string
      url: string
      short_url?: string
    }>
  }
  export default content
}
