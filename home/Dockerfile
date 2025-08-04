# This Dockerfile relies on turborepo to optimize the build process.
# Building this image requires the entire metta repo to be used as a build context, due to the global `pnpm-lock.yaml` file.

# ================================ Base image - pnpm and turbo ================================
FROM node:20-alpine AS base

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
RUN turbo prune softmax-home --docker

# =============================== Build from pruned source ================================
FROM base AS builder
WORKDIR /metta

COPY --from=pruned-source /metta/out/json/ .
RUN pnpm install --frozen-lockfile

COPY --from=pruned-source /metta/out/full/ .
# turbo's pruned sources don't include some files
COPY home ./home

WORKDIR /metta/home

# Generate nginx.conf from template and links.yaml
RUN node generate-nginx-redirects.js

# Build the application
RUN pnpm build

# ================================ Runtime stage ================================
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /metta/home/dist /usr/share/nginx/html

# Copy generated nginx configuration from builder stage
COPY --from=builder /metta/home/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]