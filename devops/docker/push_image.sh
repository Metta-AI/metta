# docker login -u mettaai
# docker push mettaai/metta:latest
# docker push mettaai/metta-base:latest

docker tag mettaai/metta:latest 767406518141.dkr.ecr.us-east-1.amazonaws.com/metta:latest
docker push 767406518141.dkr.ecr.us-east-1.amazonaws.com/metta:latest
