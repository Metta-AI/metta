# This Dockerfile relies on turborepo to optimize the build process.
# Refer to https://turborepo.com/docs/guides/tools/docker for more details.

# Building this image requires the entire metta repo to be used as a build context, due to the global `pnpm-lock.yaml` file.

# ================================ Base image - pnpm and turbo ================================
FROM node:24-bookworm-slim AS base

# Set working directory
WORKDIR /metta

ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"

# Install dependencies
COPY package.json ./
RUN corepack enable pnpm
RUN pnpm install -g turbo

# ================================ Prepare pruned source with turbo ================================
FROM base AS pruned-source
WORKDIR /metta
COPY . .

# /metta/out will contain only the relevant files
RUN turbo prune @softmax/login --docker

# =============================== Build from pruned source ================================
FROM base AS runtime
WORKDIR /metta

COPY --from=pruned-source /metta/out/json/ .
RUN pnpm install --frozen-lockfile

COPY --from=pruned-source /metta/out/full/ .
# turbo's pruned sources don't include .env files
COPY login ./login

WORKDIR /metta/login

# Install system dependencies for PDF processing and prisma
RUN apt-get update -y && apt-get install -y \
    openssl \
    ghostscript \
    graphicsmagick \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pnpm run db:generate
RUN pnpm run build

ENV NODE_ENV=production
ENV HOST=0.0.0.0

# TODO - minimize image size? https://github.com/vercel/next.js/blob/canary/examples/with-docker/Dockerfile
# TODO - separate next.js and workers images?
