#!/usr/bin/env python3
"""Convert AWS IAM secret access key to SES SMTP password.

AWS SES requires a special SMTP password format derived from the IAM secret
using the AWS Signature Version 4 signing algorithm.

Usage:
    python3 convert_ses_password.py <aws_secret_access_key>

Returns:
    JSON object with the converted password: {"password": "..."}
"""

import base64
import hashlib
import hmac
import json
import sys


def convert_secret_to_smtp_password(secret_key: str) -> str:
    """Convert AWS IAM secret key to SES SMTP password.

    Args:
        secret_key: AWS IAM secret access key

    Returns:
        SES SMTP password string
    """
    MESSAGE = "SendRawEmail"
    VERSION = 0x04

    secret = secret_key.encode("utf-8")
    signature = hmac.new(secret, MESSAGE.encode("utf-8"), hashlib.sha256).digest()
    signature_and_version = bytes([VERSION]) + signature
    smtp_password = base64.b64encode(signature_and_version).decode("utf-8")

    return smtp_password


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: convert_ses_password.py <secret_key>"}))
        sys.exit(1)

    secret_key = sys.argv[1]
    smtp_password = convert_secret_to_smtp_password(secret_key)

    # Output as JSON for Terraform external data source
    print(json.dumps({"password": smtp_password}))


if __name__ == "__main__":
    main()
