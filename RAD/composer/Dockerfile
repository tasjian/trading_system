# Use a base image with Python
FROM nvcr.io/nvidia/nemo:24.07

# Install necessary packages to add Docker repository and software-properties
# RUN apt-get update && apt-get install -y \
#     apt-utils ca-certificates curl gnupg lsb-release software-properties-common

# # Add Docker’s official GPG key
# RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# RUN install -m 0755 -d /etc/apt/keyrings
# RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
# RUN chmod a+r /etc/apt/keyrings/docker.asc

# # Set up the stable repository
# RUN echo \
#   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
#   $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 

# Set working directory
WORKDIR /dli

RUN apt-get update && apt-get install -y apt-utils ffmpeg
RUN apt-get update && apt-get install -y build-essential bash curl unzip wget git libgl1-mesa-glx libglib2.0-0
RUN apt-get update && apt-get install -y python3.11 python3-pip 
# RUN apt-get update && apt-get install -y nvidia-container-toolkit nvidia-cuda-toolkit

#Run pip dependencies
COPY composer/requirements.txt .
RUN pip3 install --upgrade pip
# RUN pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install -r requirements.txt

########################################################

# RUN pip install langchain langchain_core langchain_community wikipedia langchainhub 

# RUN pip install --upgrade tritonclient[all] nvidia-cudnn-cu12 accelerate
# ## WARNING: Latest versions of tensorflow and pytorch are intrinsically incompatable at the moment lol
# # RUN pip install --upgrade tensorflow[and-cuda] keras tensorboard 
# RUN pip install --upgrade tensorrt cuda-python 
# RUN pip install --upgrade torch==2.2.1 torchvision torchaudio lightning vllm==0.4.1 bertviz
# RUN pip install --upgrade numexpr datasets fsspec
# RUN pip install "nvidia-modelopt[all]~=0.11.0" --extra-index-url https://pypi.nvidia.com
# RUN pip install huggingface xformers huggingface_hub transformers
# # RUN pip uninstall -y transformer-engine apex

# # This thing loves to break! When you do major things with cuda, expect to flash it

# RUN pip install --upgrade ipywidgets jupyterlab jupyter-archive==3.4.0

########################################################

RUN pip3 uninstall -y pynvml && pip install --upgrade nvidia-ml-py
RUN pip3 uninstall -y transformer-engine apex
RUN pip3 uninstall -y flash-attn && pip install flash-attn
RUN pip3 install opencv-fixer==0.2.5 && python -c "from opencv_fixer import AutoFix; AutoFix()"

RUN pip3 install --upgrade ipywidgets jupyterlab==4.0.0 jupyterlab-widgets jupyter-archive jupyter-resource-usage 

COPY composer/jupyter/ /root/.jupyter
COPY composer/jupyter/bashrc /root/.bashrc
ENV PYTHONPATH=/dli/assessment/notebook_helpers
ENV SHELL /bin/bash

# COPY notebooks
WORKDIR /dli/task

# Expose port 8888 for JupyterLab
EXPOSE 8888 
EXPOSE 9012

# Start JupyterLab when the container runs
ADD composer/entrypoint.sh /usr/local/bin

ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
