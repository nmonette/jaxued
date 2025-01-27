FROM nvcr.io/nvidia/jax:24.10-py3

# Create user
ARG UID
ARG MYUSER
RUN useradd -u $UID --create-home ${MYUSER}
USER ${MYUSER}

# default workdir
WORKDIR /home/${MYUSER}/code
# RUN chown -R ${MYUSER} /home/${MYUSER}
COPY --chown=${MYUSER} --chmod=765 . .

# install from source if needed + all the requirements
USER root

# install tmux
RUN apt-get update && apt-get install -y tmux

USER ${MYUSER}

WORKDIR /home/${MYUSER}

# RUN mkdir -p /home/${MYUSER}/.local/bin /home/${MYUSER}/.local/lib/python3.10/site-packages

# Set up environment variables for local installations
ENV PATH="/home/${MYUSER}/.local/bin:$PATH"
ENV PYTHONPATH="/home/${MYUSER}/.local/lib/python3.10/site-packages:$PYTHONPATH"


RUN python3 -m pip install --user --upgrade pip
RUN python3 -m pip install --user git+https://github.com/MichaelTMatthews/Craftax.git@main
WORKDIR /home/${MYUSER}/code
RUN python3 -m pip install --user -e .

#disabling preallocation
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
#safety measures
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

#for secrets and debug
ENV WANDB_ENTITY="amacrutherford"

# WORKDIR /home/${MYUSER}/code
RUN git config --global --add safe.directory /home/${MYUSER}/code

