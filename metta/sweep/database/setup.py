#!/usr/bin/env python3
"""Setup script for initializing the worker pool database with Supabase."""

import os
import sys
import socket
import urllib.parse
import psycopg2
from psycopg2 import sql


def get_encoded_db_url():
    """Get database URL from environment or command line, with proper encoding."""
    # Priority: 1. Environment variable, 2. Command line argument, 3. Hardcoded (deprecated)

    db_url = None

    # Check environment variable first
    if 'POSTGRES_URL' in os.environ:
        db_url = os.environ['POSTGRES_URL']
        print("Using database URL from POSTGRES_URL environment variable")
    # Check command line argument
    elif len(sys.argv) > 1:
        db_url = sys.argv[1]
        print("Using database URL from command line argument")
    else:
        # Fallback to hardcoded (deprecated - will be removed)
        print("WARNING: No POSTGRES_URL environment variable or command line argument found")
        print("Using hardcoded database URL (deprecated)")
        user = "postgres"
        password = "asimplepassword#1"  # Contains # which needs encoding
        host = "db.nbiomzrrjqtbucwsxcay.supabase.co"
        port = 5432
        database = "postgres"

        # URL-encode the password to handle special characters
        encoded_password = urllib.parse.quote(password, safe='')

        # Construct the properly encoded URL
        db_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{database}"

    # Parse and validate the URL
    try:
        # Parse the database URL to extract components
        from urllib.parse import urlparse
        parsed = urlparse(db_url)

        # Extract hostname for DNS check
        hostname = parsed.hostname
        if hostname:
            print(f"Database host: {hostname}")

            # Try to resolve the hostname
            try:
                socket.gethostbyname(hostname)
                print(f"✓ Host {hostname} resolves successfully")
            except socket.gaierror as e:
                print(f"⚠️  WARNING: Cannot resolve host {hostname}")
                print(f"   Error: {e}")
                print("\n   Possible causes:")
                print("   1. Supabase project may be paused (check https://app.supabase.com)")
                print("   2. Network connectivity issues")
                print("   3. DNS resolution problems")
                print("   4. Hostname may have changed")
                print("\n   Attempting connection anyway...")

        # Hide password in output
        if parsed.password:
            safe_url = db_url.replace(parsed.password, '****')
            print(f"Database URL (password hidden): {safe_url}")
        else:
            print(f"Database URL: {db_url}")

    except Exception as e:
        print(f"Error parsing database URL: {e}")

    return db_url


def test_connection(db_url):
    """Test the database connection."""
    print("\n1. Testing database connection...")
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                print(f"✓ Connected successfully to PostgreSQL")
                print(f"  Version: {version[:60]}...")

                cursor.execute("SELECT current_database()")
                db_name = cursor.fetchone()[0]
                print(f"  Database: {db_name}")

                cursor.execute("SELECT current_user")
                user = cursor.fetchone()[0]
                print(f"  User: {user}")

        return True
    except psycopg2.OperationalError as e:
        error_msg = str(e)
        print(f"✗ Connection failed: {error_msg}")

        # Provide specific guidance based on error type
        if "could not translate host name" in error_msg:
            print("\n🔧 Troubleshooting DNS resolution issue:")
            print("   1. Check if your Supabase project is paused:")
            print("      → Go to https://app.supabase.com")
            print("      → Select your project")
            print("      → Click 'Unpause' if the project is paused")
            print("\n   2. Verify the connection string in Supabase:")
            print("      → Go to Settings > Database")
            print("      → Copy the 'Connection string' under 'Connection Pooling'")
            print("      → Update your POSTGRES_URL environment variable")
            print("\n   3. Try using the pooler connection string instead:")
            print("      → Use port 6543 instead of 5432")
            print("      → Use the 'Connection pooling' URL from Supabase dashboard")
        elif "password authentication failed" in error_msg:
            print("\n🔧 Authentication issue:")
            print("   → Verify your password is correct")
            print("   → Check if special characters in password are properly encoded")
            print("   → Get the latest connection string from Supabase dashboard")
        elif "Connection refused" in error_msg:
            print("\n🔧 Connection refused:")
            print("   → Check if the port is correct (5432 for direct, 6543 for pooled)")
            print("   → Verify firewall/network settings")
            print("   → Try the pooler connection if using direct connection")

        return False
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def create_tables(db_url):
    """Create the job queue and worker status tables."""
    print("\n2. Creating worker pool tables...")

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:
                # Check if tables already exist
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'job_queue'
                    );
                """)
                job_queue_exists = cursor.fetchone()[0]

                if job_queue_exists:
                    print("  ℹ job_queue table already exists")
                    response = input("  Drop and recreate tables? (y/n): ")
                    if response.lower() == 'y':
                        cursor.execute("DROP TABLE IF EXISTS worker_status CASCADE;")
                        cursor.execute("DROP TABLE IF EXISTS job_signals CASCADE;")
                        cursor.execute("DROP TABLE IF EXISTS job_queue CASCADE;")
                        print("  ✓ Dropped existing tables")
                    else:
                        print("  Keeping existing tables")
                        # Still ensure job_signals table exists for multi-node support
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS job_signals (
                                id SERIAL PRIMARY KEY,
                                job_id VARCHAR(255) NOT NULL,
                                job_definition JSONB NOT NULL,
                                created_at TIMESTAMP DEFAULT NOW(),
                                expires_at TIMESTAMP DEFAULT (NOW() + INTERVAL '5 minutes'),
                                CONSTRAINT unique_job_signal UNIQUE (job_id)
                            );
                        """)
                        cursor.execute("""
                            CREATE INDEX IF NOT EXISTS idx_job_signals_expires_at
                            ON job_signals(expires_at);
                        """)
                        print("  ✓ Ensured job_signals table exists")
                        return True

                # Create job_queue table
                print("  Creating job_queue table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS job_queue (
                        id SERIAL PRIMARY KEY,
                        job_id VARCHAR(255) UNIQUE NOT NULL,
                        status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'claimed', 'running', 'completed', 'failed')),
                        job_definition JSONB NOT NULL,
                        worker_id VARCHAR(255),
                        group_id VARCHAR(255),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        claimed_at TIMESTAMP WITH TIME ZONE,
                        started_at TIMESTAMP WITH TIME ZONE,
                        completed_at TIMESTAMP WITH TIME ZONE,
                        error_message TEXT,
                        retry_count INTEGER DEFAULT 0
                    );
                """)
                print("  ✓ Created job_queue table")

                # Add group_id column if table already exists (for backward compatibility)
                cursor.execute("""
                    ALTER TABLE job_queue
                    ADD COLUMN IF NOT EXISTS group_id VARCHAR(255);
                """)

                # Create indices separately
                print("  Creating indices...")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_created ON job_queue (status, created_at);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_worker_status ON job_queue (worker_id, status);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_id ON job_queue (job_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_group_status_created ON job_queue (group_id, status, created_at);")
                print("  ✓ Created indices")

                # Create worker_status table
                print("  Creating worker_status table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS worker_status (
                        worker_id VARCHAR(255) PRIMARY KEY,
                        hostname VARCHAR(255),
                        group_id VARCHAR(255),
                        last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        status VARCHAR(50) DEFAULT 'idle' CHECK (status IN ('idle', 'busy', 'offline')),
                        current_job_id VARCHAR(255),
                        started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        metadata JSONB
                    );
                """)
                print("  ✓ Created worker_status table")

                # Add group_id column if table already exists (for backward compatibility)
                cursor.execute("""
                    ALTER TABLE worker_status
                    ADD COLUMN IF NOT EXISTS group_id VARCHAR(255);
                """)
                print("  ✓ Added group_id column if needed")

                # Create the queue stats view
                print("  Creating queue_stats view...")
                cursor.execute("""
                    CREATE OR REPLACE VIEW queue_stats AS
                    SELECT
                        status,
                        COUNT(*) as count,
                        MIN(created_at) as oldest_job,
                        MAX(created_at) as newest_job,
                        AVG(EXTRACT(EPOCH FROM (COALESCE(completed_at, NOW()) - created_at))) as avg_duration_seconds
                    FROM job_queue
                    GROUP BY status;
                """)
                print("  ✓ Created queue_stats view")

                # Create job_signals table for multi-node coordination
                print("  Creating job_signals table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS job_signals (
                        id SERIAL PRIMARY KEY,
                        job_id VARCHAR(255) NOT NULL,
                        job_definition JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW(),
                        expires_at TIMESTAMP DEFAULT (NOW() + INTERVAL '5 minutes'),
                        CONSTRAINT unique_job_signal UNIQUE (job_id)
                    );
                """)
                print("  ✓ Created job_signals table")

                # Create index for efficient expiry queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_job_signals_expires_at
                    ON job_signals(expires_at);
                """)
                print("  ✓ Created job_signals index")

                # Create cleanup function for job_signals
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION cleanup_expired_signals()
                    RETURNS void AS $$
                    BEGIN
                        DELETE FROM job_signals WHERE expires_at < NOW();
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                print("  ✓ Created cleanup_expired_signals function")

                # Create cleanup function
                print("  Creating cleanup function...")
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION cleanup_old_jobs()
                    RETURNS void AS $$
                    BEGIN
                        DELETE FROM job_queue
                        WHERE status IN ('completed', 'failed')
                        AND completed_at < NOW() - INTERVAL '7 days';
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                print("  ✓ Created cleanup_old_jobs function")

                conn.commit()
                print("\n✓ All tables created successfully!")

        return True

    except Exception as e:
        print(f"✗ Failed to create tables: {e}")
        return False


def verify_setup(db_url):
    """Verify the setup is complete."""
    print("\n3. Verifying setup...")

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cursor:
                # Check tables
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('job_queue', 'worker_status')
                    ORDER BY table_name;
                """)
                tables = cursor.fetchall()
                print("  Tables found:")
                for table in tables:
                    print(f"    ✓ {table[0]}")

                # Check if we can insert a test job
                cursor.execute("""
                    INSERT INTO job_queue (job_id, job_definition)
                    VALUES ('test_setup_job', '{"test": true}')
                    ON CONFLICT (job_id) DO UPDATE
                    SET job_definition = EXCLUDED.job_definition;
                """)

                # Check if we can query it
                cursor.execute("""
                    SELECT job_id, status FROM job_queue
                    WHERE job_id = 'test_setup_job';
                """)
                test_job = cursor.fetchone()
                if test_job:
                    print(f"  ✓ Test job created: {test_job[0]} (status: {test_job[1]})")

                # Clean up test job
                cursor.execute("DELETE FROM job_queue WHERE job_id = 'test_setup_job';")
                conn.commit()
                print("  ✓ Test job cleaned up")

        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def generate_env_export(db_url):
    """Generate export command for the user."""
    print("\n4. Setup Complete!")
    print("=" * 60)
    print("Add this to your shell configuration (.bashrc, .zshrc, etc.):")
    print()
    print(f'export POSTGRES_URL="{db_url}"')
    print()
    print("Or run it now in your current shell:")
    print(f'export POSTGRES_URL="{db_url}"')
    print("=" * 60)


def main():
    """Run the setup process."""
    print("=" * 60)
    print("Worker Pool Database Setup")
    print("=" * 60)

    # Show usage if no URL is available
    if 'POSTGRES_URL' not in os.environ and len(sys.argv) <= 1:
        print("\nUsage:")
        print("  Option 1: Set environment variable")
        print("    export POSTGRES_URL='your_database_url'")
        print("    uv run python setup_worker_pool_db.py")
        print()
        print("  Option 2: Pass as command line argument")
        print("    uv run python setup_worker_pool_db.py 'your_database_url'")
        print()
        print("Get your database URL from:")
        print("  1. Go to https://app.supabase.com")
        print("  2. Select your project")
        print("  3. Go to Settings > Database")
        print("  4. Copy the 'Connection string' (use 'Connection pooling' if available)")
        print()

    # Get properly encoded URL
    db_url = get_encoded_db_url()

    # Test connection
    if not test_connection(db_url):
        print("\n❌ Could not connect to database.")
        print("\nNext steps:")
        print("1. Check your Supabase project at https://app.supabase.com")
        print("2. Ensure the project is not paused (click 'Unpause' if needed)")
        print("3. Get the latest connection string from Settings > Database")
        print("4. Update your POSTGRES_URL environment variable")
        sys.exit(1)

    # Create tables
    if not create_tables(db_url):
        print("\n❌ Failed to create tables. Please check the error messages.")
        sys.exit(1)

    # Verify setup
    if not verify_setup(db_url):
        print("\n⚠️  Setup completed but verification had issues.")

    # Show export command
    generate_env_export(db_url)

    print("\n✅ Database setup complete! You can now:")
    print("1. Launch workers: uv run ./tools/run.py worker db_url='$POSTGRES_URL' worker_id=worker-1")
    print("2. Run a sweep with dispatcher_type=REMOTE_QUEUE")
    print("3. Monitor the queue: ./metta/sweep/management/manage_pool.sh status")


if __name__ == "__main__":
    main()