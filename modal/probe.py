import modal

app = modal.App("probe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip", "zip", "curl", "ca-certificates", "ninja-build", "build-essential", "python3-pip", "nodejs")
    .pip_install("cmake>=3.28")
)


@app.function(image=image)
def probe():
    import subprocess, os
    for cmd in ["which node", "node --version", "ls -la /usr/bin/node* /usr/local/bin/node* 2>&1 | head -5",
                "echo PATH=$PATH", "cmake --version | head -1"]:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"$ {cmd}\n{r.stdout.strip()}{r.stderr.strip()}")


@app.local_entrypoint()
def main():
    probe.remote()
