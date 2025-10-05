# Login Service Deployment

This document describes how to deploy the login service to production using the existing DevOps infrastructure.

## Architecture

The login service follows the same deployment pattern as the library service:

- **Terraform**: Provisions AWS infrastructure (RDS PostgreSQL, ECR, Secrets)
- **Helm Chart**: Deploys the Kubernetes resources
- **Docker**: Containerized Next.js application
- **Domain**: `login.softmax-research.net`

## Infrastructure Components

### AWS Resources (Terraform)

Located in `/devops/tf/login/`:

- **RDS PostgreSQL**: `softmax-login-pg` database instance
- **ECR Repository**: `softmax-login` container registry
- **Security Groups**: Database access rules
- **Kubernetes Secrets**: Environment variables and credentials

### Kubernetes Resources (Helm)

Located in `/devops/charts/login/`:

- **Namespace**: `login`
- **Deployment**: Next.js frontend application
- **Service**: ClusterIP service on port 80
- **Ingress**: HTTPS routing with Let's Encrypt SSL
- **Secrets**: Environment variables from Terraform

## Deployment Process

### 1. Provision Infrastructure

```bash
cd devops/tf/login
terraform init
terraform plan
terraform apply
```

**Required Variables:**
- `oauth_secret_arn`: ARN of AWS Secrets Manager secret containing Google OAuth credentials
- `login_secrets_arn`: ARN of AWS Secrets Manager secret for additional login service secrets
- `main_s3_bucket`: S3 bucket name (if needed for future features)

### 2. Build and Push Docker Image

```bash
cd login

# Build the image
docker build -t softmax-login .

# Tag for ECR
docker tag softmax-login:latest 751442549699.dkr.ecr.us-east-1.amazonaws.com/softmax-login:latest

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 751442549699.dkr.ecr.us-east-1.amazonaws.com
docker push 751442549699.dkr.ecr.us-east-1.amazonaws.com/softmax-login:latest
```

### 3. Deploy with Helm

```bash
cd devops/charts

# Deploy all services including login
helmfile apply

# Or deploy just the login service
helmfile apply -l name=login
```

## Configuration

### Environment Variables

The service requires these environment variables (managed by Terraform/Kubernetes secrets):

**Database:**
- `DATABASE_URL`: PostgreSQL connection string

**Authentication:**
- `NEXTAUTH_SECRET`: Random secret for NextAuth.js
- `NEXTAUTH_URL`: `https://login.softmax-research.net`
- `GOOGLE_CLIENT_ID`: Google OAuth client ID
- `GOOGLE_CLIENT_SECRET`: Google OAuth client secret

**Application:**
- `DEV_MODE`: Set to `"false"` in production
- `ALLOWED_EMAIL_DOMAINS`: `"stem.ai,softmax.com"`
- `AUTH_TRUST_HOST`: `"true"` (for production HTTPS)

### Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Google+ API
3. Create OAuth 2.0 credentials
4. Set authorized redirect URI: `https://login.softmax-research.net/api/auth/callback/google`
5. Store credentials in AWS Secrets Manager

### DNS Configuration

The service will be available at `login.softmax-research.net` via:

- **External DNS**: Automatically manages Route 53 records
- **Cert Manager**: Automatically provisions Let's Encrypt SSL certificates
- **Nginx Ingress**: Routes HTTPS traffic to the service

## Database Management

### Initial Setup

After infrastructure deployment, initialize the database:

```bash
# Get database connection details from Terraform output
DATABASE_URL=$(terraform output -raw postgres_url)

# Run migrations (if needed)
cd login
DATABASE_URL=$DATABASE_URL pnpm db:push
```

### Backups

The RDS instance is configured with:
- **Backup retention**: 7 days
- **Multi-AZ**: Enabled for high availability
- **Automated backups**: Daily snapshots

## Monitoring

### Health Checks

The service provides health check endpoints:

- **Kubernetes**: Liveness/readiness probes on `/api/health`
- **Application**: Database connectivity verification

### Logs

Access logs via kubectl:

```bash
kubectl logs -n login -l app=login-frontend -f
```

## Security

### Network Security

- **Database**: Secured with security groups (temporarily open for development)
- **HTTPS**: Enforced via ingress configuration
- **Secrets**: Stored in AWS Secrets Manager and Kubernetes secrets

### Authentication Security

- **Session Strategy**: Database-backed sessions (not JWT)
- **Domain Restriction**: Only allows `stem.ai` and `softmax.com` email domains
- **CSRF Protection**: Built into NextAuth.js

## API Endpoints

The deployed service provides:

- `GET /`: Main login page
- `GET/POST /api/auth/*`: NextAuth.js authentication endpoints
- `GET /api/user`: Current user information
- `GET /api/validate`: Session validation for other services
- `POST /api/validate`: Token validation endpoint
- `GET /api/health`: Service health check
- `GET /dashboard`: Protected dashboard (demo page)

## Integration with Other Services

Other services can validate sessions by calling:

```bash
# Validate session
curl -H "Cookie: session-cookie" https://login.softmax-research.net/api/validate

# Health check
curl https://login.softmax-research.net/api/health
```

## Troubleshooting

### Common Issues

1. **Database Connection**: Check security groups and DATABASE_URL
2. **Google OAuth**: Verify redirect URIs and credentials
3. **SSL Issues**: Check cert-manager and ingress configuration
4. **Pod Crashes**: Check resource limits and environment variables

### Debug Commands

```bash
# Check deployment status
kubectl get pods -n login

# View logs
kubectl logs -n login deployment/login-frontend

# Check secrets
kubectl get secrets -n login

# Test database connection
kubectl exec -n login deployment/login-frontend -- env | grep DATABASE_URL
```