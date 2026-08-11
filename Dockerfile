FROM python:3.11-slim

WORKDIR /app

# build-essential: needed for a couple of transitive packages (e.g. some
# faiss-cpu/chromadb dependency chains) that ship without a prebuilt
# wheel for every platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first -- the default PyPI Linux wheel bundles full CUDA
# support (multiple GB of nvidia-*-cu12 packages) that this container,
# with no GPU, never uses. Installing the CPU build up front satisfies
# sentence-transformers' torch requirement below without pip pulling in
# the CUDA variant.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "serving.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
