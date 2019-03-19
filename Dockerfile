FROM ubuntu
# ...
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*
# Use an official Python runtime as a parent image
FROM python:3.6.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r ./requirements.txt
RUN pip install tensorflow-1.6.0-cp36-cp36m-linux_x86_64.whl

EXPOSE 5000
CMD ["python", "app.py"]