import subprocess
import sys
import boto3

def get_job_ip(job_name):
   batch_client = boto3.client('batch')
   ecs_client = boto3.client('ecs')
   ec2_client = boto3.client('ec2')

   # Get the list of job queues
   job_queues = batch_client.describe_job_queues()['jobQueues']

   # Iterate over the job queues to find the job
   for job_queue in job_queues:
       job_queue_name = job_queue['jobQueueName']

       # List jobs in the job queue
       jobs = batch_client.list_jobs(jobQueue=job_queue_name, jobStatus='RUNNING')['jobSummaryList']

       # Find the job with the specified name
       for job in jobs:
           if job['jobName'] == job_name:
               # Describe the job to get the task ARN and cluster
               job_desc = batch_client.describe_jobs(jobs=[job['jobId']])
               container = job_desc['jobs'][0]['container']
               task_arn = container.get('taskArn')
               cluster_arn = container.get('containerInstanceArn')

               if task_arn and cluster_arn:
                   # Extract the cluster name from the cluster ARN
                   cluster_name = cluster_arn.split('/')[1]

                   task_desc = ecs_client.describe_tasks(cluster=cluster_name, tasks=[task_arn])
                   if task_desc['tasks']:
                       container_instance_arn = task_desc['tasks'][0]['containerInstanceArn']
                       container_instance_desc = ecs_client.describe_container_instances(cluster=cluster_name, containerInstances=[container_instance_arn])
                       ec2_instance_id = container_instance_desc['containerInstances'][0]['ec2InstanceId']

                       ec2 = boto3.client('ec2')
                       instances = ec2.describe_instances(InstanceIds=[ec2_instance_id])
                       if 'PublicIpAddress' in instances['Reservations'][0]['Instances'][0]:
                           public_ip = instances['Reservations'][0]['Instances'][0]['PublicIpAddress']
                           return public_ip

   return None

def connect_to_container(ip):
   try:
       # Establish SSH connection and check if it's successful
       ssh_check_output = subprocess.check_output(f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {ip} 'echo Connected'", shell=True).decode().strip()
       if ssh_check_output != "Connected":
           raise subprocess.CalledProcessError(1, "SSH connection check failed")

       # Retrieve container ID
       container_id_output = subprocess.check_output(f"ssh -o StrictHostKeyChecking=no -t {ip} \"docker ps | grep 'daveey/metta'\"", shell=True).decode().strip()
       if container_id_output:
           container_id = container_id_output.split()[0]
           print(f"Connecting to container {container_id} on {ip}...")
           subprocess.run(f"ssh -o StrictHostKeyChecking=no -t {ip} \"docker exec -it {container_id} /bin/bash\"", shell=True)
       else:
           print(f"No container running the 'daveey/metta' image found on the instance {ip}.")
           print("Connecting to the instance directly...")
           subprocess.run(f"ssh -o StrictHostKeyChecking=no -t {ip}", shell=True)

   except subprocess.CalledProcessError as e:
       print(f"Error: {str(e)}")
       if "Connection timed out" in str(e):
           print(f"SSH connection to {ip} timed out. Please check the instance status and network connectivity.")
       elif "Connection refused" in str(e):
           print(f"SSH connection to {ip} was refused. Please check if the instance is running and accepts SSH connections.")
       else:
           print(f"An error occurred while connecting to {ip}. Please check the instance status and SSH configuration.")
       sys.exit(1)

if __name__ == '__main__':
   if len(sys.argv) != 2:
       print("Please provide a job name as an argument.")
       sys.exit(1)

   job_name = sys.argv[1]
   ip = get_job_ip(job_name)

   if ip:
       connect_to_container(ip)
   else:
       print(f"Job '{job_name}' not found or does not have a public IP.")
       sys.exit(1)
