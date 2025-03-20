# docker login -u mettaai
# docker push mettaai/metta:latest
# docker push mettaai/metta-base:latest

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 767406518141.dkr.ecr.us-east-1.amazonaws.com
docker tag mettaai/metta:latest 767406518141.dkr.ecr.us-east-1.amazonaws.com/metta:latest
docker push 767406518141.dkr.ecr.us-east-1.amazonaws.com/metta:latest
