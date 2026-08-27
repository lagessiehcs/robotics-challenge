FROM python:3.10-slim

WORKDIR /app

# Installed at build time so the image is self-contained; the repo itself is
# bind-mounted over /app at `docker run` time (see eval_runner.sh), so code
# edits and results both flow straight to the host without a rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV MPLBACKEND=Agg
ENV MPLCONFIGDIR=/tmp/matplotlib

ENTRYPOINT ["python", "eval.py"]
CMD ["--map", "maps/1/room.yaml"]
