#!/usr/bin/env python3
"""
AWS EFS VPN Setup Tool

This script automates the setup of AWS Client VPN to access an EFS volume from a Mac.
It discovers your EFS volumes and configures all necessary AWS resources.

Requirements:
- Python 3.6+
- boto3
- OpenSSL
- AWS CLI credentials configured
"""

import ipaddress
import os
import random
import subprocess
import sys

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Error: boto3 is not installed. Please install it using 'pip install boto3'")
    sys.exit(1)

# Configuration
CERTIFICATE_DIR = os.path.expanduser("~/vpn-certs")
VPN_NAME = "efs-client-vpn"
CLIENT_CIDR = "172.30.0.0/22"  # Will be validated against VPC CIDR


# Colors for terminal output
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color


class EfsVpnSetup:
    def __init__(self):
        """Initialize the EFS VPN Setup tool."""
        self.session = boto3.Session()
        self.region = self.session.region_name or "us-east-1"

        # Initialize clients
        self.ec2 = self.session.client("ec2", region_name=self.region)
        self.efs = self.session.client("efs", region_name=self.region)
        self.acm = self.session.client("acm", region_name=self.region)

        # Initialize properties
        self.efs_id = None
        self.vpc_id = None
        self.vpc_cidr = None
        self.subnet_ids = []
        self.sg_id = None
        self.vpn_endpoint_id = None
        self.server_cert_arn = None
        self.client_cert_arn = None

        # Ensure certificate directory exists
        os.makedirs(CERTIFICATE_DIR, exist_ok=True)

    def check_prerequisites(self):
        """Check that all required tools are installed."""
        print(f"{Colors.YELLOW}Checking prerequisites...{Colors.NC}")

        # Check AWS credentials
        try:
            sts = self.session.client("sts")
            sts.get_caller_identity()
            print(f"{Colors.GREEN}AWS credentials valid.{Colors.NC}")
        except Exception as e:
            print(f"{Colors.RED}AWS credentials not configured properly: {e}{Colors.NC}")
            print("Please run 'aws configure' to set up your credentials.")
            sys.exit(1)

        # Check OpenSSL
        try:
            subprocess.run(["openssl", "version"], check=True, stdout=subprocess.PIPE)
            print(f"{Colors.GREEN}OpenSSL is installed.{Colors.NC}")
        except (subprocess.SubprocessError, FileNotFoundError):
            print(f"{Colors.RED}OpenSSL is not installed. Please install it first.{Colors.NC}")
            sys.exit(1)

        print(f"{Colors.GREEN}All prerequisites met.{Colors.NC}")

    def discover_efs_volumes(self):
        """Discover EFS volumes and let user select one."""
        print(f"{Colors.YELLOW}Discovering EFS volumes in region {self.region}...{Colors.NC}")

        try:
            response = self.efs.describe_file_systems()
            filesystems = response.get("FileSystems", [])

            if not filesystems:
                print(f"{Colors.RED}No EFS volumes found in region {self.region}.{Colors.NC}")
                change_region = input(f"{Colors.YELLOW}Would you like to try another region? (y/n) {Colors.NC}").lower()

                if change_region == "y":
                    # List available regions
                    regions = [region["RegionName"] for region in self.ec2.describe_regions()["Regions"]]
                    print(f"{Colors.YELLOW}Available regions:{Colors.NC}")
                    for i, region in enumerate(regions, 1):
                        print(f"{i}. {region}")

                    region_num = int(input(f"{Colors.YELLOW}Enter region number: {Colors.NC}"))
                    self.region = regions[region_num - 1]

                    # Reinitialize clients with new region
                    self.ec2 = self.session.client("ec2", region_name=self.region)
                    self.efs = self.session.client("efs", region_name=self.region)
                    self.acm = self.session.client("acm", region_name=self.region)

                    # Try again with new region
                    return self.discover_efs_volumes()
                else:
                    print(f"{Colors.RED}Cannot proceed without an EFS volume.{Colors.NC}")
                    sys.exit(1)

            # Display EFS volumes
            print(f"{Colors.YELLOW}Found {len(filesystems)} EFS volumes:{Colors.NC}")
            for i, fs in enumerate(filesystems, 1):
                name = fs.get("Name", "No Name")
                creation_time = fs["CreationTime"].strftime("%Y-%m-%d %H:%M:%S")
                print(f"{i}. {fs['FileSystemId']} - Name: {name} - Creation: {creation_time}")

            if len(filesystems) == 1:
                # Only one EFS, use it automatically
                self.efs_id = filesystems[0]["FileSystemId"]
                print(f"{Colors.GREEN}Using the only available EFS: {self.efs_id}{Colors.NC}")
            else:
                # Multiple EFS volumes, let user choose
                efs_num = int(input(f"{Colors.YELLOW}Enter the number of the EFS volume you want to use: {Colors.NC}"))
                self.efs_id = filesystems[efs_num - 1]["FileSystemId"]

            # Get EFS mount targets
            self.get_efs_mount_targets()

        except Exception as e:
            print(f"{Colors.RED}Error discovering EFS volumes: {str(e)}{Colors.NC}")
            sys.exit(1)

    def get_efs_mount_targets(self):
        """Get mount targets for the selected EFS volume."""
        print(f"{Colors.YELLOW}Getting EFS mount targets for {self.efs_id}...{Colors.NC}")

        try:
            response = self.efs.describe_mount_targets(FileSystemId=self.efs_id)
            mount_targets = response.get("MountTargets", [])

            if not mount_targets:
                print(f"{Colors.RED}No mount targets found for EFS {self.efs_id}.{Colors.NC}")
                print(f"{Colors.RED}Please create at least one mount target first.{Colors.NC}")
                sys.exit(1)

            # Extract VPC ID from mount targets
            self.vpc_id = mount_targets[0]["VpcId"]
            print(f"{Colors.GREEN}Discovered VPC: {self.vpc_id}{Colors.NC}")

            # Extract subnet IDs from mount targets
            self.subnet_ids = [mt["SubnetId"] for mt in mount_targets]
            print(f"{Colors.GREEN}Discovered subnet IDs: {', '.join(self.subnet_ids)}{Colors.NC}")

            # Get mount targets details for display
            print(f"{Colors.YELLOW}EFS mount targets:{Colors.NC}")
            for mt in mount_targets:
                print(f"{mt['MountTargetId']} - Subnet: {mt['SubnetId']} - IP: {mt['IpAddress']}")

            # Get VPC CIDR
            vpc_response = self.ec2.describe_vpcs(VpcIds=[self.vpc_id])
            self.vpc_cidr = vpc_response["Vpcs"][0]["CidrBlock"]
            print(f"{Colors.GREEN}VPC CIDR: {self.vpc_cidr}{Colors.NC}")

            # Check if CLIENT_CIDR overlaps with VPC_CIDR
            self.check_cidr_overlap()

        except Exception as e:
            print(f"{Colors.RED}Error getting EFS mount targets: {str(e)}{Colors.NC}")
            sys.exit(1)

    def check_cidr_overlap(self):
        """Check if CLIENT_CIDR overlaps with VPC_CIDR."""
        global CLIENT_CIDR

        try:
            vpc_network = ipaddress.IPv4Network(self.vpc_cidr)
            client_network = ipaddress.IPv4Network(CLIENT_CIDR)

            if vpc_network.overlaps(client_network):
                print(
                    f"{Colors.RED}Warning: Client CIDR ({CLIENT_CIDR}) overlaps with VPC CIDR ({self.vpc_cidr}).{Colors.NC}"
                )
                new_cidr = input(f"{Colors.YELLOW}Please enter a new Client CIDR (e.g., 172.31.0.0/22): {Colors.NC}")
                CLIENT_CIDR = new_cidr
                # Recursive check with new CIDR
                self.check_cidr_overlap()
            else:
                print(
                    f"{Colors.GREEN}Client CIDR ({CLIENT_CIDR}) does not overlap with VPC CIDR ({self.vpc_cidr}).{Colors.NC}"
                )

        except ValueError as e:
            print(f"{Colors.RED}Invalid CIDR format: {str(e)}{Colors.NC}")
            new_cidr = input(f"{Colors.YELLOW}Please enter a valid CIDR (e.g., 172.31.0.0/22): {Colors.NC}")
            CLIENT_CIDR = new_cidr
            self.check_cidr_overlap()

    def generate_certificates(self):
        """Generate certificates for mutual authentication."""
        print(f"{Colors.YELLOW}Generating certificates for mutual authentication...{Colors.NC}")

        os.chdir(CERTIFICATE_DIR)

        try:
            # Generate CA key and certificate
            subprocess.run(["openssl", "genrsa", "-out", "ca.key", "2048"], check=True)
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-new",
                    "-x509",
                    "-days",
                    "3650",
                    "-key",
                    "ca.key",
                    "-out",
                    "ca.crt",
                    "-subj",
                    "/CN=EFS VPN CA",
                ],
                check=True,
            )

            # Generate server key and certificate
            subprocess.run(["openssl", "genrsa", "-out", "server.key", "2048"], check=True)
            subprocess.run(
                ["openssl", "req", "-new", "-key", "server.key", "-out", "server.csr", "-subj", "/CN=server"],
                check=True,
            )

            # Create server certificate extension file
            with open("server.ext", "w") as f:
                f.write("""basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:server
""")

            # Sign server certificate
            subprocess.run(
                [
                    "openssl",
                    "x509",
                    "-req",
                    "-days",
                    "3650",
                    "-in",
                    "server.csr",
                    "-CA",
                    "ca.crt",
                    "-CAkey",
                    "ca.key",
                    "-CAcreateserial",
                    "-out",
                    "server.crt",
                    "-extfile",
                    "server.ext",
                ],
                check=True,
            )

            # Generate client key and certificate
            subprocess.run(["openssl", "genrsa", "-out", "client.key", "2048"], check=True)
            subprocess.run(
                ["openssl", "req", "-new", "-key", "client.key", "-out", "client.csr", "-subj", "/CN=client"],
                check=True,
            )

            # Create client certificate extension file
            with open("client.ext", "w") as f:
                f.write("""basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=clientAuth
subjectAltName=DNS:client
""")

            # Sign client certificate
            subprocess.run(
                [
                    "openssl",
                    "x509",
                    "-req",
                    "-days",
                    "3650",
                    "-in",
                    "client.csr",
                    "-CA",
                    "ca.crt",
                    "-CAkey",
                    "ca.key",
                    "-CAcreateserial",
                    "-out",
                    "client.crt",
                    "-extfile",
                    "client.ext",
                ],
                check=True,
            )

            # Import certificates to ACM
            print(f"{Colors.YELLOW}Importing certificates to AWS Certificate Manager...{Colors.NC}")

            # Read certificate files
            with open("server.crt", "rb") as f:
                server_cert = f.read()
            with open("server.key", "rb") as f:
                server_key = f.read()
            with open("ca.crt", "rb") as f:
                ca_cert = f.read()
            with open("client.crt", "rb") as f:
                client_cert = f.read()
            with open("client.key", "rb") as f:
                client_key = f.read()

            # Import server certificate
            response = self.acm.import_certificate(
                Certificate=server_cert, PrivateKey=server_key, CertificateChain=ca_cert
            )
            self.server_cert_arn = response["CertificateArn"]
            print(f"{Colors.GREEN}Server certificate imported with ARN: {self.server_cert_arn}{Colors.NC}")

            # Import client certificate
            response = self.acm.import_certificate(
                Certificate=client_cert, PrivateKey=client_key, CertificateChain=ca_cert
            )
            self.client_cert_arn = response["CertificateArn"]
            print(f"{Colors.GREEN}Client certificate imported with ARN: {self.client_cert_arn}{Colors.NC}")

            # Create initial client configuration (will be updated later)
            with open("client-config.ovpn", "w") as f:
                f.write(f"""client
dev tun
proto udp
remote cvpn-endpoint-{random.randint(1000, 9999)}.clientvpn.{self.region}.amazonaws.com 443
remote-random-hostname
resolv-retry infinite
nobind
persist-key
persist-tun
remote-cert-tls server
cipher AES-256-GCM
verb 3
<ca>
{open("ca.crt", "r").read()}
</ca>
<cert>
{open("client.crt", "r").read()}
</cert>
<key>
{open("client.key", "r").read()}
</key>
reneg-sec 0
""")

        except subprocess.SubprocessError as e:
            print(f"{Colors.RED}Error generating certificates: {str(e)}{Colors.NC}")
            sys.exit(1)

    def create_security_group(self):
        """Create security group for VPN endpoint."""
        print(f"{Colors.YELLOW}Creating security group for VPN endpoint...{Colors.NC}")

        try:
            # Check if security group already exists
            existing_sg = None
            try:
                response = self.ec2.describe_security_groups(
                    Filters=[
                        {"Name": "group-name", "Values": [f"{VPN_NAME}-sg"]},
                        {"Name": "vpc-id", "Values": [self.vpc_id]},
                    ]
                )
                if response["SecurityGroups"]:
                    existing_sg = response["SecurityGroups"][0]["GroupId"]
            except ClientError:
                pass

            if existing_sg:
                print(f"{Colors.GREEN}Using existing security group: {existing_sg}{Colors.NC}")
                self.sg_id = existing_sg
            else:
                # Create security group
                response = self.ec2.create_security_group(
                    GroupName=f"{VPN_NAME}-sg",
                    Description=f"Security group for {VPN_NAME} Client VPN",
                    VpcId=self.vpc_id,
                )
                self.sg_id = response["GroupId"]

                print(f"{Colors.GREEN}Created security group: {self.sg_id}{Colors.NC}")

                # Add tags
                self.ec2.create_tags(Resources=[self.sg_id], Tags=[{"Key": "Name", "Value": f"{VPN_NAME}-sg"}])

                # Allow outbound traffic
                print(f"{Colors.YELLOW}Adding outbound rule to security group...{Colors.NC}")
                self.ec2.authorize_security_group_egress(
                    GroupId=self.sg_id, IpPermissions=[{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}]
                )

            # Update EFS security groups to allow access from VPN
            self.update_efs_security_groups()

        except ClientError as e:
            if "InvalidPermission.Duplicate" in str(e):
                print(f"{Colors.YELLOW}Outbound rule already exists (this is OK){Colors.NC}")
            else:
                print(f"{Colors.RED}Error creating security group: {str(e)}{Colors.NC}")
                sys.exit(1)

    def update_efs_security_groups(self):
        """Update EFS security groups to allow access from VPN."""
        try:
            # Get EFS mount targets
            response = self.efs.describe_mount_targets(FileSystemId=self.efs_id)

            for mt in response["MountTargets"]:
                # Get security groups for mount target
                sg_response = self.efs.describe_mount_target_security_groups(MountTargetId=mt["MountTargetId"])
                efs_sg = sg_response["SecurityGroups"][0]

                print(
                    f"{Colors.YELLOW}Updating EFS security group {efs_sg} to allow NFS access from VPN security group...{Colors.NC}"
                )

                # Check if rule already exists
                try:
                    self.ec2.describe_security_groups(
                        GroupIds=[efs_sg],
                        Filters=[
                            {"Name": "ip-permission.from-port", "Values": ["2049"]},
                            {"Name": "ip-permission.to-port", "Values": ["2049"]},
                            {"Name": "ip-permission.protocol", "Values": ["tcp"]},
                            {"Name": "ip-permission.group-id", "Values": [self.sg_id]},
                        ],
                    )
                    print(f"{Colors.GREEN}NFS access rule already exists in security group {efs_sg}{Colors.NC}")
                except ClientError:
                    # Rule doesn't exist, add it
                    try:
                        self.ec2.authorize_security_group_ingress(
                            GroupId=efs_sg,
                            IpPermissions=[
                                {
                                    "IpProtocol": "tcp",
                                    "FromPort": 2049,
                                    "ToPort": 2049,
                                    "UserIdGroupPairs": [{"GroupId": self.sg_id}],
                                }
                            ],
                        )
                        print(f"{Colors.GREEN}Added NFS access rule to security group {efs_sg}{Colors.NC}")
                    except ClientError as e:
                        if "InvalidPermission.Duplicate" in str(e):
                            print(f"{Colors.YELLOW}Rule already exists (this is OK){Colors.NC}")
                        else:
                            raise

        except Exception as e:
            print(f"{Colors.RED}Error updating EFS security groups: {str(e)}{Colors.NC}")
            sys.exit(1)

    def create_vpn_endpoint(self):
        """Create the Client VPN endpoint."""
        print(f"{Colors.YELLOW}Creating Client VPN endpoint...{Colors.NC}")

        try:
            # Check if VPN endpoint already exists
            existing_vpn = None
            try:
                response = self.ec2.describe_client_vpn_endpoints(Filters=[{"Name": "tag:Name", "Values": [VPN_NAME]}])
                if response["ClientVpnEndpoints"]:
                    existing_vpn = response["ClientVpnEndpoints"][0]["ClientVpnEndpointId"]
            except ClientError:
                pass

            if existing_vpn:
                print(f"{Colors.GREEN}Using existing VPN endpoint: {existing_vpn}{Colors.NC}")
                self.vpn_endpoint_id = existing_vpn
            else:
                # Create Client VPN endpoint
                response = self.ec2.create_client_vpn_endpoint(
                    ClientCidrBlock=CLIENT_CIDR,
                    ServerCertificateArn=self.server_cert_arn,
                    AuthenticationOptions=[
                        {
                            "Type": "certificate-authentication",
                            "MutualAuthentication": {"ClientRootCertificateChainArn": self.client_cert_arn},
                        }
                    ],
                    ConnectionLogOptions={"Enabled": False},
                    Description=f"{VPN_NAME} for EFS access",
                    SecurityGroupIds=[self.sg_id],
                    SplitTunnel=True,
                    TransportProtocol="udp",
                    VpcId=self.vpc_id,
                    TagSpecifications=[
                        {"ResourceType": "client-vpn-endpoint", "Tags": [{"Key": "Name", "Value": VPN_NAME}]}
                    ],
                )

                self.vpn_endpoint_id = response["ClientVpnEndpointId"]
                print(f"{Colors.GREEN}Created Client VPN endpoint: {self.vpn_endpoint_id}{Colors.NC}")

                # Wait for the endpoint to become available
                print(f"{Colors.YELLOW}Waiting for VPN endpoint to become available...{Colors.NC}")
                waiter = self.ec2.get_waiter("client_vpn_endpoint_available")
                waiter.wait(ClientVpnEndpointIds=[self.vpn_endpoint_id])

            # Get DNS name for client configuration
            response = self.ec2.describe_client_vpn_endpoints(ClientVpnEndpointIds=[self.vpn_endpoint_id])
            vpn_dns = response["ClientVpnEndpoints"][0]["DnsName"]

            # Update client configuration with the correct DNS name
            with open(os.path.join(CERTIFICATE_DIR, "client-config.ovpn"), "r") as f:
                config = f.read()

            config = config.replace(
                f"remote cvpn-endpoint-{random.randint(1000, 9999)}.clientvpn.{self.region}.amazonaws.com 443",
                f"remote {vpn_dns} 443",
            )

            with open(os.path.join(CERTIFICATE_DIR, "client-config.ovpn"), "w") as f:
                f.write(config)

            print(f"{Colors.GREEN}Updated client configuration with VPN endpoint DNS: {vpn_dns}{Colors.NC}")

        except Exception as e:
            print(f"{Colors.RED}Error creating VPN endpoint: {str(e)}{Colors.NC}")
            sys.exit(1)
