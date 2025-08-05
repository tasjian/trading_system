# Use a base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /dli

RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y build-essential bash curl unzip wget git libgl1-mesa-glx libglib2.0-0 

#Run pip dependencies
COPY composer/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --upgrade ipywidgets jupyterlab==4.3.5 jupyterlab-widgets jupyter-archive jupyter-resource-usage 
RUN pip install uv

WORKDIR /dli/task

# Copy a jupyterlab workspace into the image to set initial jupyter lab view. See `set-landing` for details.
COPY jupyter/ /root/.jupyter

# Start JupyterLab when the container runs
ADD composer/entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
