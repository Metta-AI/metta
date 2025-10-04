# Login Service

A dedicated authentication service extracted from the library service, using the same tech stack.

## Features

- **NextAuth.js v5** - Modern authentication for Next.js
- **Google OAuth** - Production authentication provider
- **Fake Email Provider** - For development (logs magic links to console)
- **Prisma** - Type-safe database access
- **PostgreSQL** - Database backend
- **Domain-based Email Filtering** - Restrict access to specific email domains
- **Session Management** - Database-backed sessions
- **API Endpoints** - Validation and user info endpoints

## Tech Stack

- Next.js 15.3.1
- TypeScript 5
- NextAuth.js 5.0.0-beta.29
- Prisma 6.12.0
- PostgreSQL
- Tailwind CSS 4
- React 19

## Getting Started

### Prerequisites

- Node.js 18+
- PostgreSQL database
- Google OAuth credentials (for production)

### Installation

```bash
pnpm install
```

### Environment Setup

Create a `.env.local` file:

```env
# Database
DATABASE_URL="postgresql://username:password@localhost:5432/login_db"

# NextAuth.js
NEXTAUTH_SECRET="your-secret-here"
NEXTAUTH_URL="http://localhost:3002"

# Development mode (enables fake email provider)
DEV_MODE="true"

# Allowed email domains (comma-separated)
ALLOWED_EMAIL_DOMAINS="stem.ai,softmax.com"

# Google OAuth (production)
GOOGLE_CLIENT_ID="your-google-client-id"
GOOGLE_CLIENT_SECRET="your-google-client-secret"
```

### Database Setup

```bash
# Generate Prisma client
pnpm db:generate

# Push schema to database
pnpm db:push

# Or run migrations
pnpm db:migrate
```

### Development

```bash
# Start development server on port 3002
pnpm dev
```

The service will be available at http://localhost:3002

## API Endpoints

### Authentication
- `GET/POST /api/auth/*` - NextAuth.js endpoints
- `GET /api/auth/signin` - Sign in page
- `GET /api/auth/signout` - Sign out

### User Management
- `GET /api/user` - Get current user information
- `GET /api/validate` - Validate current session
- `POST /api/validate` - Validate session token (for other services)

### Health Check
- `GET /api/health` - Service health and database connectivity

## Usage in Other Services

Other services can validate sessions by calling:

```typescript
// Validate current session
const response = await fetch('http://localhost:3002/api/validate');
const { valid, user } = await response.json();

// Validate specific session token
const response = await fetch('http://localhost:3002/api/validate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ sessionToken: 'token-here' })
});
```

## Development vs Production

### Development Mode (`DEV_MODE=true`)
- Fake email provider available
- Magic links logged to console
- Less strict domain validation

### Production Mode (`DEV_MODE=false` or unset)
- Only Google OAuth available
- Strict email domain validation
- Secure session handling

## Database Schema

The service uses these auth-related tables:
- `user` - User accounts
- `account` - OAuth provider accounts
- `session` - Active sessions
- `verificationToken` - Email verification tokens
- `authenticator` - WebAuthn authenticators

## Security Features

- Database-backed sessions
- Email domain restrictions
- Secure OAuth flow
- CSRF protection
- Session validation API for microservices