import css from '@eslint/css'
import eslint from '@eslint/js'
import tseslint from 'typescript-eslint'
import { globalIgnores } from 'eslint/config'

export default tseslint.config(
  globalIgnores(['dist', 'index.html', 'index.css']),
  {
    name: 'eslint',
    files: ['**/*.ts'],
    ...eslint.configs.recommended,
  },
  {
    name: 'typescript-eslint',
    files: ['**/*.ts'],
    extends: tseslint.configs.recommended,
  },
  {
    files: ['style.css'], // no point in linting index.css, it's auto-generated
    plugins: { css },
    language: 'css/css',
    extends: [css.configs.recommended],
  },
  // disable some rules
  {
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
      'no-prototype-builtins': 'off',
      'css/use-baseline': 'off',
      'css/no-important': 'off',
    },
  }
)
