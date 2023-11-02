FROM gitpod/workspace-full

RUN pyenv install 3.11.6 \
    && pyenv global 3.11.6
