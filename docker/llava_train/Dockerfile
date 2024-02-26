FROM smellslikeml/vqasynth:base
WORKDIR /app

RUN git clone https://github.com/haotian-liu/LLaVA.git 
WORKDIR /app/LLaVA

RUN pip install -e ".[train]"
RUN pip install flash-attn --no-build-isolation

COPY entrypoint.sh /app/LLaVA/
RUN chmod +x /app/LLaVA/entrypoint.sh

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/LLaVA/entrypoint.sh"]
