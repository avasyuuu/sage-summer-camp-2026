FROM waggle/sage-thor-base:0.1.0

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Publish to the beehive by default: on a node that is the only useful output,
# since the container's local files disappear and an off-node watcher is what
# turns these measurements into SMS/Slack alerts. Extra args from the job spec
# are appended to this command.
ENTRYPOINT ["python3", "scripts/main.py", "--publish"]
