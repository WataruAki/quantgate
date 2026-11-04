FROM gcc:14-bookworm
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
COPY worker.py digits.py /opt/
COPY digits-data/ /opt/digits-data/
USER 65534:65534
WORKDIR /opt
ENTRYPOINT ["python3", "-I", "-c", "import sys; sys.path.insert(0, '/opt'); import worker; worker.main()"]
