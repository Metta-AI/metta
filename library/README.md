# Softmax Library

A social feed and knowledge repository for AI research papers, built with Next.js 15, Prisma ORM, and NextAuth.js.

## Development

### Prerequisites

1. **PostgreSQL Database**: You need to run [Postgres.app](https://postgresapp.com/) locally
   - Download and install Postgres.app
   - Start the PostgreSQL server
   - Create a database named `metta_library` (or update the DATABASE_URL in your .env.local)

2. **Node.js**: Ensure you have Node.js installed (version 18 or higher recommended)

### Setup

1. **Install dependencies**:
   ```bash
   pnpm install
   ```

2. **Set up environment variables**:
   Create a `.env.local` file with the following content:
   ```
   DATABASE_URL=postgres://localhost/metta_library
   DEV_MODE=true
   ```

3. **Generate authentication secret**:
   ```bash
   pnpm auth secret
   ```
   This will populate your `.env.local` file with a random `AUTH_SECRET`.

4. **Set up the database**:
   ```bash
   # Generate Prisma client
   pnpm prisma generate
   
   # Run database migrations (if any)
   pnpm prisma db push
   ```

### Running

1. **Start the development server**:
   ```bash
   pnpm dev
   ```

2. **Access the application**:
   Open your browser and go to `http://localhost:3001`

3. **Sign in**:
   - Use any email address to sign in
   - Check the terminal logs for the magic link
   - Click the magic link to complete the sign in process

### Database Management

- **View database**: Use `pnpm prisma studio` to open Prisma Studio and browse your data
- **Reset database**: Use `pnpm prisma db push --force-reset` to reset the database (⚠️ this will delete all data)
- **Generate client**: Use `pnpm prisma generate` after schema changes

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

### Authentication

Authentication is handled by [NextAuth.js](https://next-auth.js.org/) with the Prisma adapter.

- **Development**: Uses a fake email provider that logs magic links to the console
- **Production**: Configure your preferred email provider in the auth configuration

See [src/lib/auth.ts](src/lib/auth.ts) for more details.

### Database

The application uses [Prisma ORM](https://www.prisma.io/) for database access:

- **Schema**: Defined in `prisma/schema.prisma`
- **Client**: Generated automatically with `pnpm prisma generate`
- **Migrations**: Applied with `pnpm prisma db push`

The database schema includes:
- **Users**: Authentication and user profiles
- **Papers**: Research papers with metadata
- **Posts**: User-generated content and discussions
- **Interactions**: User interactions with papers (stars, queue)

### Pagination

The application implements cursor-based pagination for efficient data loading:

- [src/posts/data/feed.ts](src/posts/data/feed.ts) - Feed data access with pagination support
- [src/lib/hooks/usePaginator.ts](src/lib/hooks/usePaginator.ts) - React hook for managing pagination state
- [src/components/LoadMore.tsx](src/components/LoadMore.tsx) - UI component for loading more items
- [src/components/InfiniteScroll.tsx](src/components/InfiniteScroll.tsx) - Infinite scroll implementation

### Available Scripts

- `pnpm dev` - Start development server
- `pnpm build` - Build for production
- `pnpm start` - Start production server
- `pnpm lint` - Run linting
- `pnpm format` - Format code with Prettier
- `pnpm prisma studio` - Open Prisma Studio for database management
- `pnpm prisma generate` - Generate Prisma client
- `pnpm prisma db push` - Push schema changes to database
