FROM waggle/sage-thor-base:0.1.0

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Publish to the beehive by default: on a node that is the only useful output,
# since the container's local files disappear and an off-node watcher is what
# turns these measurements into SMS/Slack alerts. Extra args from the job spec
# are appended to this command.
#
# --state-file matters on Sage: the scheduler re-runs this plugin as a fresh
# process on every cron tick, so without persisted tracker state an animal that
# stays in view is re-published as a new animal each time. For this to work the
# path must survive between invocations - mount a persistent volume at /app/data.
ENV TRACKER_STATE_FILE=/app/data/tracker_state.json
ENTRYPOINT ["python3", "scripts/main.py", "--publish"]
