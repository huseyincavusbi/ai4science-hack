# Science of AI / ML Track Starting Kit

Submit a zip with `run.sh` at the root.

Codabench will call:

```bash
./run.sh --task /path/to/task --output /path/to/predictions.csv --output-dir /path/to/output
```

Your script must write `predictions.csv` with columns `id,prediction`.
The included baseline zip is a lightweight pure-Python baseline intended
only to smoke-test the submission path.

`public_packets/train_tasks/` contains labeled practice packets.
`public_packets/validation_task/` contains the public validation inputs.
Validation labels and final labels stay server-side.
