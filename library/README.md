# Softmax Library

## Development

### Setup

Run `pnpm install` to install dependencies.

Create a `.env.local` file with the following content:

```
DATABASE_URL=postgres://localhost/metta_library
DEV_MODE=true
```

Then run `pnpx auth secret` to populate this file with a random `AUTH_SECRET`.

### Running

Run `pnpm dev` to start the development server.

Then go to `http://localhost:3001` to see the app.

Use any email to sign in; click the logged magic link in the terminal logs to complete the sign in.

### Code layout

- `src/app` - Next.js app directory
- `src/components` - shared components that need to be used on more than one page
- `src/lib` - common `*.ts` files (no React components); some of these can be server-only
- `src/lib/hooks` - React hooks

Most of other code should be organized under `src/{entity}` dirs, with one dir per "content type", for example, "posts", "comments", "users", etc.

For example, `src/posts` contains:

- `src/posts/actions` - server-side actions
- `src/posts/data` - data access layer

If you need to reuse content type-specific React components, you should put them under `src/{entity}/components` dir.

### Auth

Auth is handled by [Auth.js](https://authjs.dev/).

In development, we use a fake email provider that logs the magic link to the console.

See [src/lib/auth.ts](src/lib/auth.ts) for more details.

### Pagination

Example implementation:

- [src/posts/data/feed.ts](src/posts/data/feed.ts) for selecting a feed that supports infinite scrolling
- [src/lib/hooks/usePaginator.ts](src/lib/hooks/usePaginator.ts) for a React hook that manages the pagination state
- [src/components/LoadMore.tsx](src/components/LoadMore.tsx) for a component that can be used to load more items
