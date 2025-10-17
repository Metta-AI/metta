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
   # Core application settings
   DATABASE_URL=postgresql://localhost/metta_library
   DEV_MODE=true
   NEXTAUTH_SECRET=your-generated-secret
   NEXTAUTH_URL=http://localhost:3001
   ALLOWED_EMAIL_DOMAINS=stem.ai,softmax.com

   # Discord integration (optional)
   DISCORD_CLIENT_ID=your-discord-client-id
   DISCORD_CLIENT_SECRET=your-discord-client-secret
   DISCORD_REDIRECT_URI=http://localhost:3001/api/auth/callback/discord
   NEXT_PUBLIC_DISCORD_CLIENT_ID=your-discord-client-id

   # Google OAuth (optional, overrides DEV_MODE fake email provider)
   GOOGLE_CLIENT_ID=your-google-client-id
   GOOGLE_CLIENT_SECRET=your-google-client-secret

   # Email notifications (optional)
   ENABLE_EMAIL_NOTIFICATIONS=true
   EMAIL_FROM_ADDRESS=notifications@yourapp.com
   EMAIL_FROM_NAME=Library Notifications

   # AWS SES configuration (optional)
   AWS_SES_ACCESS_KEY_ID=your-aws-ses-access-key
   AWS_SES_SECRET_ACCESS_KEY=your-aws-ses-secret
   AWS_SES_REGION=us-east-1

   # SMTP fallback configuration (optional if SES not used)
   SMTP_HOST=smtp.sendgrid.net
   SMTP_PORT=587
   SMTP_USER=apikey
   SMTP_PASSWORD=your-smtp-password

   # Redis / BullMQ job queue
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_PASSWORD=
   REDIS_TLS=false
   ```

# AWS miscellaneous (optional for PDF processing)

AWS_REGION=us-east-1
AWS_PROFILE=softmax
AWS_S3_BUCKET=metta-pdf-processing

# LLM integrations (optional)

ANTHROPIC_API_KEY=your-anthropic-key

# Adobe PDF Services (optional)

ADOBE_CLIENT_ID=your-adobe-client-id
ADOBE_CLIENT_SECRET=your-adobe-client-secret

# Figure extraction (optional, disabled by default)

# Set to "true" to enable figure image and text extraction from PDFs

# Note: Figure extraction is currently experimental and may not work reliably

ENABLE_FIGURE_EXTRACTION=false
NEXT_PUBLIC_ENABLE_FIGURE_EXTRACTION=false

# Asana integration (optional)

ASANA_API_KEY=
ASANA_TOKEN=
ASANA_PAPERS_PROJECT_ID=
ASANA_WORKSPACE_ID=
ASANA_PAPER_LINK_FIELD_ID=
ASANA_ARXIV_ID_FIELD_ID=
ASANA_ABSTRACT_FIELD_ID=

````

> Only populate the sections relevant to the features you plan to exercise locally. Leave optional values blank to disable the corresponding integration.

3. **Generate authentication secret**:

```bash
pnpm auth secret
````

This will populate your `.env.local` file with a random `NEXTAUTH_SECRET`.

4. **Set up the database**:

   ```bash
   # Generate Prisma client
   pnpm prisma generate

   # Create a development migration baseline
   pnpm prisma migrate dev --name init
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
- `pnpm prisma migrate dev` - Create & apply schema migrations locally
- `pnpm fetch-arxiv` - Fetch arXiv paper metadata as JSON
- `pnpm test-arxiv` - Test the arXiv fetcher module

### arXiv Paper Fetcher

The project includes a reusable script for fetching arXiv paper metadata as JSON. This script can be used both as a command-line tool and imported as a module in your code.

#### Command Line Usage

```bash
# Fetch paper by URL
pnpm fetch-arxiv https://arxiv.org/abs/2204.11674

# Fetch paper by arXiv ID
pnpm fetch-arxiv 2204.11674
```

#### Module Usage

```typescript
import { fetchArxivPaper, ArxivPaperData } from "./scripts/fetch-arxiv-paper";

// Fetch paper data
const paperData: ArxivPaperData = await fetchArxivPaper("2204.11674");

// Use the structured data
console.log(paperData.title);
console.log(paperData.authors);
console.log(paperData.abstract);
```

#### Output Format

The script returns a structured JSON object with the following fields:

- `id`: arXiv ID (e.g., "2204.11674")
- `title`: Paper title
- `abstract`: Paper abstract
- `authors`: Array of author names
- `categories`: Array of arXiv categories
- `publishedDate`: Publication date (ISO string)
- `updatedDate`: Last update date (ISO string)
- `doi`: DOI if available
- `journalRef`: Journal reference if available
- `primaryCategory`: Primary arXiv category
- `arxivUrl`: arXiv abstract URL
- `pdfUrl`: Direct PDF download URL
- `summary`: Alias for abstract

#### Testing

Run the test script to verify the module works correctly:

```bash
pnpm test-arxiv
```
