/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_EVAL_DB_URI: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
