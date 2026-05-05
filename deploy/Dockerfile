FROM rayproject/ray:2.52.0

WORKDIR /app


USER root
RUN apt-get update && apt-get install -y \
build-essential \
&& rm -rf /var/lib/apt/lists/*

USER ray

COPY serve_requirements.txt /app
RUN pip install --no-cache-dir -r serve_requirements.txt

COPY src /app/src
