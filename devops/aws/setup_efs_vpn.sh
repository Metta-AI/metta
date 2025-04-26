#!/bin/bash
#
# AWS Client VPN Setup for EFS Access
# This script automatically discovers your EFS volumes and sets up a VPN to access them from a Mac
#
# Prerequisites:
# - AWS CLI installed and configured with appropriate permissions
# - openssl installed for certificate generation
# - jq installed for JSON parsing

set -e

# Configuration parameters (modify only if needed)
REGION=$(aws configure get region || echo "us-east-1")  # Get region from AWS config or default
CLIENT_CIDR="172.30.0.0/22"       # CIDR for VPN clients (automatically checked for overlaps)
CERTIFICATE_DIR="$HOME/vpn-certs" # Directory to store certificates
VPN_NAME="efs-client-vpn"         # Name for your VPN endpoint

# Variables to be discovered automatically
VPC_ID=""                         # Will be discovered based on EFS mount targets
SUBNET_IDS=""                     # Will be discovered based on EFS mount targets
EFS_ID=""                         # Will be discovered or selected by user

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
        echo "Visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        exit 1
    fi
    
    # Check AWS CLI configuration
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}AWS CLI is not configured properly. Please run 'aws configure'.${NC}"
        exit 1
    fi
    
    # Check openssl
    if ! command -v openssl &> /dev/null; then
        echo -e "${RED}openssl is not installed. Please install it first.${NC}"
        exit 1
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}jq is not installed. Please install it first.${NC}"
        echo "For macOS: brew install jq"
        exit 1
    fi
    
    echo -e "${GREEN}All prerequisites met.${NC}"
}

# Function to discover EFS volumes and select one
discover_efs_volumes() {
    echo -e "${YELLOW}Discovering EFS volumes in region $REGION...${NC}"
    
    # Get list of EFS volumes
    EFS_LIST=$(aws efs describe-file-systems --region $REGION)
    EFS_COUNT=$(echo $EFS_LIST | jq '.FileSystems | length')
    
    if [ "$EFS_COUNT" -eq "0" ]; then
        echo -e "${RED}No EFS volumes found in region $REGION.${NC}"
        echo -e "${YELLOW}Would you like to try another region? (y/n)${NC}"
        read change_region
        
        if [[ $change_region == "y" || $change_region == "Y" ]]; then
            echo -e "${YELLOW}Available regions:${NC}"
            aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output table
            echo -e "${YELLOW}Enter region name:${NC}"
            read REGION
            discover_efs_volumes
            return
        else
            echo -e "${RED}Cannot proceed without an EFS volume.${NC}"
            exit 1
        fi
    fi
    
    # Display EFS volumes
    echo -e "${YELLOW}Found $EFS_COUNT EFS volumes:${NC}"
    echo $EFS_LIST | jq -r '.FileSystems[] | "\(.FileSystemId) - Name: \(.Name // "No Name") - Creation: \(.CreationTime)"' | nl
    
    if [ "$EFS_COUNT" -eq "1" ]; then
        # Only one EFS, use it automatically
        EFS_ID=$(echo $EFS_LIST | jq -r '.FileSystems[0].FileSystemId')
        echo -e "${GREEN}Using the only available EFS: $EFS_ID${NC}"
    else
        # Multiple EFS volumes, let user choose
        echo -e "${YELLOW}Enter the number of the EFS volume you want to use:${NC}"
        read efs_number
        EFS_ID=$(echo $EFS_LIST | jq -r ".FileSystems[$(($efs_number-1))].FileSystemId")
    fi
    
    # Get EFS mount targets
    echo -e "${YELLOW}Getting EFS mount targets for $EFS_ID...${NC}"
    MOUNT_TARGETS=$(aws efs describe-mount-targets --file-system-id $EFS_ID --region $REGION)
    MOUNT_TARGET_COUNT=$(echo $MOUNT_TARGETS | jq '.MountTargets | length')
    
    if [ "$MOUNT_TARGET_COUNT" -eq "0" ]; then
        echo -e "${RED}No mount targets found for EFS $EFS_ID.${NC}"
        echo -e "${RED}Please create at least one mount target first.${NC}"
        exit 1
    fi
    
    # Extract VPC ID from mount targets
    VPC_ID=$(echo $MOUNT_TARGETS | jq -r '.MountTargets[0].VpcId')
    echo -e "${GREEN}Discovered VPC: $VPC_ID${NC}"
    
    # Extract subnet IDs from mount targets
    SUBNET_IDS=$(echo $MOUNT_TARGETS | jq -r '.MountTargets[].SubnetId' | tr '\n' ',' | sed 's/,$//')
    echo -e "${GREEN}Discovered subnet IDs: $SUBNET_IDS${NC}"
    
    # Get mount targets details for display
    echo -e "${YELLOW}EFS mount targets:${NC}"
    echo $MOUNT_TARGETS | jq -r '.MountTargets[] | "\(.MountTargetId) - Subnet: \(.SubnetId) - IP: \(.IpAddress)"'
    
    # Get VPC CIDR for authorization rule
    VPC_CIDR=$(aws ec2 describe-vpcs --vpc-ids $VPC_ID --region $REGION --query 'Vpcs[0].CidrBlock' --output text)
    echo -e "${GREEN}VPC CIDR: $VPC_CIDR${NC}"
    
    # Check if CLIENT_CIDR overlaps with VPC_CIDR
    check_cidr_overlap
}

# Function to check if CLIENT_CIDR overlaps with VPC_CIDR
check_cidr_overlap() {
    # Extract VPC CIDR parts
    VPC_IP=$(echo $VPC_CIDR | cut -d'/' -f1)
    VPC_PREFIX=$(echo $VPC_CIDR | cut -d'/' -f2)
    
    # Extract Client CIDR parts
    CLIENT_IP=$(echo $CLIENT_CIDR | cut -d'/' -f1)
    CLIENT_PREFIX=$(echo $CLIENT_CIDR | cut -d'/' -f2)
    
    # Convert IP to integer
    ip_to_int() {
        local IFS='.'
        read -r i1 i2 i3 i4 <<< "$1"
        echo $(( (i1 << 24) + (i2 << 16) + (i3 << 8) + i4 ))
    }
    
    # Calculate network address
    network_address() {
        local ip=$1
        local prefix=$2
        local mask=$(( 0xFFFFFFFF << (32 - prefix) ))
        echo $(( $(ip_to_int "$ip") & mask ))
    }
    
    # Calculate broadcast address
    broadcast_address() {
        local ip=$1
        local prefix=$2
        local mask=$(( 0xFFFFFFFF << (32 - prefix) ))
        echo $(( $(ip_to_int "$ip") | ~mask & 0xFFFFFFFF ))
    }
    
    VPC_NETWORK=$(network_address "$VPC_IP" "$VPC_PREFIX")
    VPC_BROADCAST=$(broadcast_address "$VPC_IP" "$VPC_PREFIX")
    CLIENT_NETWORK=$(network_address "$CLIENT_IP" "$CLIENT_PREFIX")
    CLIENT_BROADCAST=$(broadcast_address "$CLIENT_IP" "$CLIENT_PREFIX")
    
    # Check for overlap
    if [ "$CLIENT_NETWORK" -le "$VPC_BROADCAST" ] && [ "$CLIENT_BROADCAST" -ge "$VPC_NETWORK" ]; then
        echo -e "${RED}Warning: Client CIDR ($CLIENT_CIDR) overlaps with VPC CIDR ($VPC_CIDR).${NC}"
        echo -e "${YELLOW}Please enter a new Client CIDR (e.g., 172.31.0.0/22):${NC}"
        read CLIENT_CIDR
        check_cidr_overlap
    else
        echo -e "${GREEN}Client CIDR ($CLIENT_CIDR) does not overlap with VPC CIDR ($VPC_CIDR).${NC}"
    fi
}

# Function to generate certificates
generate_certificates() {
    echo -e "${YELLOW}Generating certificates for mutual authentication...${NC}"
    
    mkdir -p $CERTIFICATE_DIR
    cd $CERTIFICATE_DIR
    
    # Generate CA key and certificate
    openssl genrsa -out ca.key 2048
    openssl req -new -x509 -days 3650 -key ca.key -out ca.crt -subj "/CN=EFS VPN CA"
    
    # Generate server key and certificate
    openssl genrsa -out server.key 2048
    openssl req -new -key server.key -out server.csr -subj "/CN=server"
    
    # Create server certificate extension file
    cat > server.ext << EOF
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:server
EOF
    
    # Sign server certificate
    openssl x509 -req -days 3650 -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -extfile server.ext
    
    # Generate client key and certificate
    openssl genrsa -out client.key 2048
    openssl req -new -key client.key -out client.csr -subj "/CN=client"
    
    # Create client certificate extension file
    cat > client.ext << EOF
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=clientAuth
subjectAltName=DNS:client
EOF
    
    # Sign client certificate
    openssl x509 -req -days 3650 -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -extfile client.ext
    
    # Import the certificates to ACM
    echo -e "${YELLOW}Importing certificates to AWS Certificate Manager...${NC}"
    
    # Import server certificate
    SERVER_CERT_ARN=$(aws acm import-certificate --certificate fileb://server.crt \
                       --private-key fileb://server.key \
                       --certificate-chain fileb://ca.crt \
                       --region $REGION --query 'CertificateArn' --output text)
    
    echo -e "${GREEN}Server certificate imported with ARN: $SERVER_CERT_ARN${NC}"
    
    # Import client certificate
    CLIENT_CERT_ARN=$(aws acm import-certificate --certificate fileb://client.crt \
                       --private-key fileb://client.key \
                       --certificate-chain fileb://ca.crt \
                       --region $REGION --query 'CertificateArn' --output text)
    
    echo -e "${GREEN}Client certificate imported with ARN: $CLIENT_CERT_ARN${NC}"
    
    # Create client configuration
    cat > client-config.ovpn << EOF
client
dev tun
proto udp
remote cvpn-endpoint-$(echo $RANDOM).clientvpn.$REGION.amazonaws.com 443
remote-random-hostname
resolv-retry infinite
nobind
persist-key
persist-tun
remote-cert-tls server
cipher AES-256-GCM
verb 3
<ca>
$(cat ca.crt)
</ca>
<cert>
$(cat client.crt)
</cert>
<key>
$(cat client.key)
</key>
reneg-sec 0
EOF
}

# Function to create security group for VPN endpoint
create_security_group() {
    echo -e "${YELLOW}Creating security group for VPN endpoint...${NC}"
    
    # Check if security group already exists
    EXISTING_SG=$(aws ec2 describe-security-groups \
                  --filters "Name=group-name,Values=$VPN_NAME-sg" "Name=vpc-id,Values=$VPC_ID" \
                  --region $REGION \
                  --query 'SecurityGroups[0].GroupId' \
                  --output text 2>/dev/null)
    
    if [ "$EXISTING_SG" != "None" ] && [ -n "$EXISTING_SG" ]; then
        echo -e "${GREEN}Using existing security group: $EXISTING_SG${NC}"
        SG_ID=$EXISTING_SG
    else
        # Create security group
        SG_ID=$(aws ec2 create-security-group \
                --group-name "$VPN_NAME-sg" \
                --description "Security group for $VPN_NAME Client VPN" \
                --vpc-id $VPC_ID \
                --region $REGION \
                --query 'GroupId' --output text)
        
        echo -e "${GREEN}Created security group: $SG_ID${NC}"
        
        # Add tags
        aws ec2 create-tags --resources $SG_ID --tags "Key=Name,Value=$VPN_NAME-sg" --region $REGION
        
        # Allow outbound traffic
        echo -e "${YELLOW}Adding outbound rule to security group...${NC}"
        aws ec2 authorize-security-group-egress \
            --group-id $SG_ID \
            --ip-permissions '[{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}]' \
            --region $REGION || echo -e "${YELLOW}Outbound rule already exists (this is OK)${NC}"
    fi
        
    # Get EFS security group(s)
    EFS_MOUNT_TARGETS=$(aws efs describe-mount-targets --file-system-id $EFS_ID --region $REGION --query 'MountTargets[*].MountTargetId' --output text)
    
    for MT_ID in $EFS_MOUNT_TARGETS; do
        EFS_SG=$(aws efs describe-mount-target-security-groups --mount-target-id $MT_ID --region $REGION --query 'SecurityGroups[0]' --output text)
        
        echo -e "${YELLOW}Updating EFS security group $EFS_SG to allow NFS access from VPN security group...${NC}"
        
        # Check if rule already exists
        RULE_EXISTS=$(aws ec2 describe-security-groups \
                      --group-ids $EFS_SG \
                      --region $REGION \
                      --filters "Name=ip-permission.from-port,Values=2049" \
                                "Name=ip-permission.to-port,Values=2049" \
                                "Name=ip-permission.protocol,Values=tcp" \
                                "Name=ip-permission.group-id,Values=$SG_ID" \
                      --query 'SecurityGroups[0].GroupId' \
                      --output text 2>/dev/null)
        
        if [ "$RULE_EXISTS" != "None" ] && [ -n "$RULE_EXISTS" ]; then
            echo -e "${GREEN}NFS access rule already exists in security group $EFS_SG${NC}"
        else
            # Allow NFS access from VPN security group to EFS security group
            aws ec2 authorize-security-group-ingress \
                --group-id $EFS_SG \
                --protocol tcp \
                --port 2049 \
                --source-group $SG_ID \
                --region $REGION || echo -e "${YELLOW}Rule already exists (this is OK)${NC}"
        fi
    done
    
    echo $SG_ID
}

# Function to create the Client VPN endpoint
create_vpn_endpoint() {
    echo -e "${YELLOW}Creating Client VPN endpoint...${NC}"
    
    # Check if VPN endpoint already exists
    EXISTING_VPN=$(aws ec2 describe-client-vpn-endpoints \
                  --filters "Name=tag:Name,Values=$VPN_NAME" \
                  --region $REGION \
                  --query 'ClientVpnEndpoints[0].ClientVpnEndpointId' \
                  --output text 2>/dev/null)
    
    if [ "$EXISTING_VPN" != "None" ] && [ -n "$EXISTING_VPN" ]; then
        echo -e "${GREEN}Using existing VPN endpoint: $EXISTING_VPN${NC}"
        VPN_ENDPOINT_ID=$EXISTING_VPN
    else
        # Create Client VPN endpoint
        VPN_ENDPOINT_ID=$(aws ec2 create-client-vpn-endpoint \
                          --client-cidr-block $CLIENT_CIDR \
                          --server-certificate-arn $SERVER_CERT_ARN \
                          --authentication-options Type=certificate-authentication,MutualAuthentication={ClientRootCertificateChainArn=$CLIENT_CERT_ARN} \
                          --connection-log-options Enabled=false \
                          --description "$VPN_NAME for EFS access" \
                          --security-group-ids $SG_ID \
                          --split-tunnel \
                          --transport-protocol udp \
                          --vpc-id $VPC_ID \
                          --region $REGION \
                          --query 'ClientVpnEndpointId' --output text)
        
        echo -e "${GREEN}Created Client VPN endpoint: $VPN_ENDPOINT_ID${NC}"
        
        # Add tags
        aws ec2 create-tags --resources $VPN_ENDPOINT_ID --tags "Key=Name,Value=$VPN_NAME" --region $REGION
        
        # Wait for the endpoint to become available
        echo -e "${YELLOW}Waiting for VPN endpoint to become available...${NC}"
        aws ec2 wait client-vpn-endpoint-available --client-vpn-endpoint-id $VPN_ENDPOINT_ID --region $REGION
    fi
    
    # Get DNS name for client configuration
    VPN_DNS=$(aws ec2 describe-client-vpn-endpoints \
              --client-vpn-endpoint-id $VPN_ENDPOINT_ID \
              --region $REGION \
              --query 'ClientVpnEndpoints[0].DnsName' --output text)
    
    # Update client configuration with the correct DNS name
    sed -i.bak "s/cvpn-endpoint-.*\.clientvpn\.$REGION\.amazonaws\.com/$VPN_DNS/g" $CERTIFICATE_DIR/client-config.ovpn
    
    echo $VPN_ENDPOINT_ID
}

# Function to associate VPN endpoint with subnets
associate_vpn_subnets() {
    echo -e "${YELLOW}Associating VPN endpoint with subnets...${NC}"
    
    IFS=',' read -ra SUBNET_ARRAY <<< "$SUBNET_IDS"
    
    for SUBNET_ID in "${SUBNET_ARRAY[@]}"; do
        echo -e "${YELLOW}Checking association with subnet $SUBNET_ID...${NC}"
        
        # Check if association already exists
        EXISTING_ASSOC=$(aws ec2 describe-client-vpn-target-networks \
                         --client-vpn-endpoint-id $VPN_ENDPOINT_ID \
                         --region $REGION \
                         --filters "Name=target-network-id,Values=$SUBNET_ID" \
                         --query 'ClientVpnTargetNetworks[0].AssociationId' \
                         --output text 2>/dev/null)
        
        if [ "$EXISTING_ASSOC" != "None" ] && [ -n "$EXISTING_ASSOC" ]; then
            echo -e "${GREEN}Subnet $SUBNET_ID is already associated with VPN endpoint${NC}"
            continue
        fi
        
        echo -e "${YELLOW}Associating with subnet $SUBNET_ID...${NC}"
        
        ASSOCIATION_ID=$(aws ec2 associate-client-vpn-target-network \
                         --client-vpn-endpoint-id $VPN_ENDPOINT_ID \
                         --subnet-id $SUBNET_ID \
                         --region $REGION \
                         --query 'AssociationId' --output text)
        
        echo -e "${GREEN}Created association: $ASSOCIATION_ID${NC}"
        
        # Wait for the association to complete
        echo -e "${YELLOW}Waiting for association to complete...${NC}"
        aws ec2 wait client-vpn-target-network-association-available \
            --client-vpn-endpoint-id $VPN_ENDPOINT_ID \
            --association-id $ASSOCIATION_ID \
            --region $REGION
    done
}

# Function to authorize VPN access to VPC
authorize_vpn_access() {
    echo -e "${YELLOW}Authorizing VPN access to VPC resources...${NC}"
    
    # Check if authorization rule already exists
    EXISTING_AUTH=$(aws ec2 describe-client-vpn-authorization-rules \
                    --client-vpn-endpoint-id $VPN_ENDPOINT_ID \
                    --region $REGION \
                    --filters "Name=destination-cidr,Values=$VPC_CIDR" \
                    --query 'AuthorizationRules[0].DestinationCidr' \
                    --output text 2>/dev/null)
    
    if [ "$EXISTING_AUTH" != "None" ] && [ -n "$EXISTING_AUTH" ]; then
        echo -e "${GREEN}Authorization rule for $VPC_CIDR already exists${NC}"
    else
        # Authorize access to VPC CIDR
        aws ec2 authorize-client-vpn-ingress \
            --client-vpn-endpoint-id $VPN_ENDPOINT_ID \
            --target-network-cidr $VPC_CIDR \
            --authorize-all-groups \
            --region $REGION
        
        echo -e "${GREEN}Authorized access to VPC CIDR: $VPC_CIDR${NC}"
    fi
    
    # Add route for internet access if it doesn't exist
    for SUBNET_ID in "${SUBNET_ARRAY[@]}"; do
        echo -e "${YELLOW}Checking route for subnet $SUBNET_ID...${NC}"
        
        # Check if route already exists
        EXISTING_ROUTE=$(aws ec2 describe-client-vpn-routes \
                        --client-vpn-endpoint-id $VPN_ENDPOINT_ID \
                        --region $REGION \
                        --filters "Name=destination-cidr,Values=0.0.0.0/0" "Name=target-subnet,Values=$SUBNET_ID" \
                        --query 'Routes[0].DestinationCidr' \
                        --output text 2>/dev/null)
        
        if [ "$EXISTING_ROUTE" != "None" ] && [ -n "$EXISTING_ROUTE" ]; then
            echo -e "${GREEN}Route for 0.0.0.0/0 through subnet $SUBNET_ID already exists${NC}"
            break  # Only need one route for internet access
        else
            echo -e "${YELLOW}Adding route for internet access through subnet $SUBNET_ID...${NC}"
            
            aws ec2 create-client-vpn-route \
                --client-vpn-endpoint-id $VPN_ENDPOINT_ID \
                --destination-cidr-block 0.0.0.0/0 \
                --target-vpc-subnet-id $SUBNET_ID \
                --region $REGION || echo -e "${YELLOW}Failed to add route, trying another subnet...${NC}"
                
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Added route for internet access through subnet $SUBNET_ID${NC}"
                break  # Successfully added a route
            fi
        fi
    done
}

# Function to create a CloudFormation template for easier cleanup
create_cloudformation_template() {
    echo -e "${YELLOW}Creating CloudFormation template for VPN resources...${NC}"
    
    cat > $CERTIFICATE_DIR/vpn-cloudformation.yaml << EOF
AWSTemplateFormatVersion: '2010-09-09'
Description: 'AWS Client VPN for EFS Access'
Resources:
  ClientVpnEndpoint:
    Type: AWS::EC2::ClientVpnEndpoint
    Properties:
      ClientCidrBlock: $CLIENT_CIDR
      ServerCertificateArn: $SERVER_CERT_ARN
      AuthenticationOptions:
        - Type: certificate-authentication
          MutualAuthentication:
            ClientRootCertificateChainArn: $CLIENT_CERT_ARN
      ConnectionLogOptions:
        Enabled: false
      Description: $VPN_NAME for EFS access
      DnsServers:
        - 8.8.8.8
        - 8.8.4.4
      SecurityGroupIds:
        - $SG_ID
      SplitTunnel: true
      TransportProtocol: udp
      VpcId: $VPC_ID
      TagSpecifications:
        - ResourceType: client-vpn-endpoint
          Tags:
            - Key: Name
              Value: $VPN_NAME
EOF

    # Add associations
    IFS=',' read -ra SUBNET_ARRAY <<< "$SUBNET_IDS"
    
    for i in "${!SUBNET_ARRAY[@]}"; do
        cat >> $CERTIFICATE_DIR/vpn-cloudformation.yaml << EOF
  ClientVpnTargetNetwork$i:
    Type: AWS::EC2::ClientVpnTargetNetworkAssociation
    Properties:
      ClientVpnEndpointId: !Ref ClientVpnEndpoint
      SubnetId: ${SUBNET_ARRAY[$i]}
EOF
    done

    # Add authorization rule
    cat >> $CERTIFICATE_DIR/vpn-cloudformation.yaml << EOF
  ClientVpnAuthorizationRule:
    Type: AWS::EC2::ClientVpnAuthorizationRule
    Properties:
      ClientVpnEndpointId: !Ref ClientVpnEndpoint
      TargetNetworkCidr: $VPC_CIDR
      AuthorizeAllGroups: true
EOF

    # Add route
    cat >> $CERTIFICATE_DIR/vpn-cloudformation.yaml << EOF
  ClientVpnRoute:
    Type: AWS::EC2::ClientVpnRoute
    DependsOn: ClientVpnTargetNetwork0
    Properties:
      ClientVpnEndpointId: !Ref ClientVpnEndpoint
      DestinationCidrBlock: 0.0.0.0/0
      TargetVpcSubnetId: ${SUBNET_ARRAY[0]}
Outputs:
  ClientVpnEndpointId:
    Description: Client VPN Endpoint ID
    Value: !Ref ClientVpnEndpoint
  ClientVpnEndpointDNS:
    Description: Client VPN Endpoint DNS Name
    Value: !GetAtt ClientVpnEndpoint.DnsName
EOF

    echo -e "${GREEN}CloudFormation template created at:${NC} $CERTIFICATE_DIR/vpn-cloudformation.yaml"
    echo -e "${YELLOW}You can use this template to manage or delete the resources later.${NC}"
}

# Main function
main() {
    echo -e "${GREEN}=== AWS Client VPN Setup for EFS Access ===${NC}"
    
    check_prerequisites
    discover_efs_volumes
    generate_certificates
    SG_ID=$(create_security_group)
    VPN_ENDPOINT_ID=$(create_vpn_endpoint)
    associate_vpn_subnets
    authorize_vpn_access
    create_cloudformation_template
    
    echo -e "${GREEN}=== Setup Complete ===${NC}"
    echo -e "${YELLOW}Client VPN configuration file is available at:${NC} $CERTIFICATE_DIR/client-config.ovpn"
    echo -e "${YELLOW}To connect to the VPN:${NC}"
    echo "1. Install the AWS VPN Client for macOS: https://aws.amazon.com/vpn/client-vpn-download/"
    echo "2. Import the client configuration file"
    echo "3. Connect to the VPN"
    
    echo -e "${YELLOW}To mount the EFS volume after connecting to VPN:${NC}"
    echo "sudo mkdir -p /path/to/mount/point"
    echo "sudo mount -t nfs -o vers=4,tcp $EFS_ID.efs.$REGION.amazonaws.com:/ /path/to/mount/point"
    
    echo -e "${YELLOW}For automatic mounting on your Mac, add this line to /etc/fstab:${NC}"
    echo "$EFS_ID.efs.$REGION.amazonaws.com:/ /path/to/mount/point nfs vers=4,tcp,rw,auto,_netdev 0 0"
    
    echo -e "${RED}Important: This script has created AWS resources that may incur charges.${NC}"
    echo -e "${RED}Client VPN endpoints cost approximately \$0.10-\$0.15 per hour.${NC}"
    echo -e "${YELLOW}To delete these resources, you can use the CloudFormation template created at:${NC}"
    echo "$CERTIFICATE_DIR/vpn-cloudformation.yaml"
}

# Show script summary
echo -e "${GREEN}=== AWS Client VPN Setup for EFS Access ===${NC}"
echo "This script will:"
echo "1. Automatically discover your EFS volumes"
echo "2. Set up AWS Client VPN to access your EFS from your Mac"
echo "3. Generate all necessary certificates and configuration files"
echo "4. Create a CloudFormation template for easy resource management"
echo -e "${YELLOW}NOTE: This will create AWS resources that may incur charges.${NC}"
echo -e "${YELLOW}Do you want to continue? (y/n)${NC}"
read confirm

if [[ $confirm == "y" || $confirm == "Y" ]]; then
    main
else
    echo "Script cancelled."
    exit 0
fi