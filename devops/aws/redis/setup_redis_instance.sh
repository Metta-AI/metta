#!/bin/bash
# save as setup_redis_docker.sh
set -e

echo "Setting up Redis with Docker on AWS..."

# Configuration
INSTANCE_TYPE="t3.micro" # x86 for better Docker compatibility
REGION="us-east-1"
REDIS_PASSWORD=$(openssl rand -base64 32)  # Auto-generate password
SG_ID="sg-0589d6aca4e16d184" # Reuse existing security group

echo "Generated Redis password: $REDIS_PASSWORD"

# Get Amazon Linux 2 AMI (x86)
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters \
        "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text \
    --region $REGION)

echo "Using AMI: $AMI_ID"

# Create user data script
cat > /tmp/docker-redis.sh << EOF
#!/bin/bash
# Log all output
exec > /var/log/user-data.log 2>&1
set -x

# Update and install Docker
yum update -y
amazon-linux-extras install docker -y
service docker start
systemctl enable docker

# Pull and run Redis
docker pull redis:7-alpine
docker run -d \
  --name redis \
  --restart always \
  -p 6379:6379 \
  redis:7-alpine \
  redis-server --requirepass "$REDIS_PASSWORD" --bind 0.0.0.0 --protected-mode yes

# Wait for Redis to be ready
sleep 10

# Check if Redis is running
docker ps
docker logs redis

echo "Setup complete"
EOF

# Launch instance
echo "Launching EC2 instance with Docker..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --security-group-ids $SG_ID \
    --user-data file:///tmp/docker-redis.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=redis-docker-lock-server}]" \
    --query 'Instances[0].InstanceId' \
    --output text \
    --region $REGION)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to start..."

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
    --region $REGION)

# Clean up temp file
rm /tmp/docker-redis.sh

echo "Instance IP: $PUBLIC_IP"
echo "Waiting 60 seconds for Docker and Redis to start..."

# Progress bar
for i in {1..60}; do
    echo -n "."
    sleep 1
done
echo ""

# Test connection
echo "Testing Redis connection..."
for i in {1..10}; do
    if redis-cli -h $PUBLIC_IP -p 6379 -a "$REDIS_PASSWORD" ping 2>&1 | grep -q PONG; then
        echo "✅ Redis is ready!"
        break
    fi
    echo "Attempt $i/10..."
    sleep 3
done

# Create redis.env file
cat > redis.env << ENV_FILE
REDIS_HOST=$PUBLIC_IP
REDIS_PORT=6379
REDIS_PASSWORD=$REDIS_PASSWORD
INSTANCE_ID=$INSTANCE_ID
REGION=$REGION
SECURITY_GROUP_ID=$SG_ID
ENV_FILE

# Final connection info
echo ""
echo "========================================="
echo "Redis Docker Server Setup Complete!"
echo "========================================="
echo ""
echo "Connection Details:"
echo "  Host: $PUBLIC_IP"
echo "  Port: 6379"
echo "  Password: $REDIS_PASSWORD"
echo ""
echo "Test connection:"
echo "  redis-cli -h $PUBLIC_IP -p 6379 -a '$REDIS_PASSWORD' ping"
echo ""
echo "Python test:"
echo "  python3 -c \"import redis; r=redis.Redis('$PUBLIC_IP',6379,'$REDIS_PASSWORD'); print(r.ping())\""
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Region: $REGION"
echo "Security Group: $SG_ID"
echo ""
echo "To stop instance: aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION"
echo "To terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo "========================================="
echo ""
echo "✅ Created redis.env file with connection details"
echo ""
echo "⚠️  Remember to add 'redis.env' to your .gitignore file!"
echo ""
echo "Usage in Python:"
echo "  from metta.util.redis_lock import redis_lock"
echo "  with redis_lock('my_resource'):"
echo "      # Your code here"
echo ""
echo "The redis_lock will automatically load credentials from redis.env"
