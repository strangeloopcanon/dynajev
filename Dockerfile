# CPU image. For CUDA, swap the base for a pytorch/pytorch:*-cuda* image and
# drop the --index-url line; the code picks the device automatically.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    DYNAJEV_MODEL=Qwen/Qwen3.5-2B DYNAJEV_HOST=0.0.0.0 DYNAJEV_PORT=43124

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install .

# Pull the weights at build time so the container starts without network access.
# Comment this out to fetch lazily on first start instead.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-2B', allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','*.jinja'])"

EXPOSE 43124
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:43124/api/ready', timeout=2).status==200 else 1)"
CMD ["dynajev", "serve"]
