# Cleaning up

None of this is required. If you plan to keep experimenting, keep everything — the model is the
slow part to get back, and `uv run agentgraph doctor` will still pass tomorrow.

But this workshop pulled a multi-gigabyte model onto your machine, possibly installed Ollama and
`uv`, and may have written a variable into your shell profile. If you want your laptop back
exactly as it was, here is every piece of it.

The model is the **Thinking** checkpoint: a separate 8 GB download from the Instruct one
`agentfix-langchain` uses, sharing no blobs with it. If you worked through both editions, both are
on disk and each has to be removed by name.

**On Option 3 (Colab) there is nothing to clean on this machine.** The model, the Ollama install
and the clone all lived in the Colab runtime, which is discarded when the session ends. Delete the
notebook copy from your Drive if Colab saved one there, and you are done.

### What is on your machine, and what put it there

| What | Created by | Where it lives | Size |
|---|---|---|---|
| `agentgraph-mellum2-thinking` + its base model | Option 1 | Ollama's model store | ~8 GB |
| `qwen3:1.7b` | Option 2 | Ollama's model store | ~1.4 GB |
| `MELLUM_MODEL` in your shell profile | Option 2, only if you put it there | `~/.zshrc`, `~/.bashrc`, or the Windows user environment | — |
| `.venv` and `uv.lock`-resolved packages | `uv sync` | inside the clone | a few hundred MB |
| uv's package cache | `uv sync` | `~/.cache/uv` (`%LOCALAPPDATA%\uv` on Windows) | up to a few hundred MB |
| `agentgraph-sandbox` Docker image | only if you built it | Docker | a few hundred MB |
| Ollama | only if you installed it for this | per OS, see below | a few hundred MB |
| `uv`, and a uv-installed Python 3.12 | only if you installed them for this | `~/.local/bin`, `~/.local/share/uv` | ~50 MB |
| Run results | `agentgraph solve` / `eval` | `results/` in the clone | KBs |

Nothing was installed into your system Python: the environment lives inside the clone, so deleting
the folder takes the packages with it.

**Nothing in the repository was generated.** `Modelfile`, `Dockerfile.sandbox` and
`results/precomputed/` all ship with it — there is no setup script here that wrote files into the
repo root, so there is nothing to delete except the clone itself.

Open your operating system below. Each block is the complete sequence in the reverse of the order
setup did it; skip the steps that do not apply to you.

<details>
<summary><b>macOS</b></summary>

**1. Remove the model** — undoes `ollama pull` + `ollama create`. This is almost all of the disk
space.

```bash
ollama list                                                             # see what you have
```

```bash
# Option 1
ollama rm agentgraph-mellum2-thinking hf.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M
```

```bash
# Option 2
ollama rm qwen3:1.7b
```

Remove the derived model **and** the base model it was built from. The base model is the
multi-gigabyte download; deleting only `agentgraph-mellum2-thinking` frees almost nothing.

**2. Remove the model variable** — only if you set `MELLUM_MODEL` for Option 2.

If you put it in `~/.zshrc` (or `~/.bashrc`, or `~/.config/fish/config.fish`), delete that line.
Then, for the terminal you are standing in right now:

```bash
unset MELLUM_MODEL
```

**3. Remove Ollama itself** — optional, and only if you installed it for this workshop. Do the
model first: uninstalling does not reliably take the model store with it.

```bash
# installed with Homebrew
brew services stop ollama
brew uninstall ollama
```

If you installed the app instead: quit Ollama from the menu bar, then drag **Ollama.app** from
**Applications** to the Trash.

Either way, the leftovers neither route removes:

```bash
rm -rf ~/.ollama
sudo rm -f /usr/local/bin/ollama
```

If you set the loaded-model cap while juggling two editions, undo that too:

```bash
launchctl unsetenv OLLAMA_MAX_LOADED_MODELS
```

**4. Remove the Docker sandbox image** — only if you built it.

```bash
docker rmi agentgraph-sandbox
```

**5. Remove `uv` and its cache** — only if you have no other use for it. A `brew`-installed `uv`
is a normal package.

```bash
uv cache clean
uv python uninstall 3.12                                       # only a uv-installed one
brew uninstall uv                                              # if you installed it that way
rm -rf ~/.local/share/uv ~/.local/bin/uv ~/.local/bin/uvx      # if you used the install script
```

**6. Delete the clone**

```bash
rm -rf ~/agentfix-react          # wherever you cloned it
```

That takes `.venv`, `results/` and everything you wrote with it. Copy out any solution you want to
keep first.

**7. Check it worked**

```bash
ollama list                          # no agentgraph-… entry
echo "$MELLUM_MODEL"                 # empty
```
</details>

<details>
<summary><b>Linux, WSL2 and ChromeOS</b></summary>

**1. Remove the model** — undoes `ollama pull` + `ollama create`. This is almost all of the disk
space.

```bash
ollama list                                                             # see what you have
```

```bash
# Option 1
ollama rm agentgraph-mellum2-thinking hf.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M
```

```bash
# Option 2
ollama rm qwen3:1.7b
```

Remove the derived model **and** the base model it was built from — the base model is the
multi-gigabyte download.

**2. Remove the model variable** — only if you set `MELLUM_MODEL` for Option 2. Delete the line
from `~/.bashrc` (or `~/.zshrc`, or `~/.config/fish/config.fish`), then:

```bash
unset MELLUM_MODEL
```

**3. Remove Ollama itself** — optional, and only if you installed it for this workshop. Do the
model first. The install script registers a systemd service, so stop that before deleting the
binary:

```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm -f /etc/systemd/system/ollama.service
sudo systemctl daemon-reload

sudo rm -f "$(command -v ollama)"
rm -rf ~/.ollama
```

If you added `OLLAMA_MAX_LOADED_MODELS=1` with `systemctl edit ollama`, that drop-in went with the
service file above. Inside WSL2 without systemd you started the server with `ollama serve`, so
there is no service to remove — stop that process, then run the last two commands.

The installer also creates a dedicated `ollama` user with its own model store. Only if nothing else
on the machine uses them:

```bash
sudo rm -rf /usr/share/ollama
sudo userdel ollama
sudo groupdel ollama
```

**4. Remove the Docker sandbox image** — only if you built it.

```bash
docker rmi agentgraph-sandbox
```

**5. Remove `uv` and its cache** — only if you have no other use for it. An `apt`/`dnf`/`pacman`
`uv` is a normal package: uninstall it with the same tool.

```bash
uv cache clean
uv python uninstall 3.12                                       # only a uv-installed one
rm -rf ~/.local/share/uv ~/.local/bin/uv ~/.local/bin/uvx      # if you used the install script
```

**6. Delete the clone**

```bash
rm -rf ~/agentfix-react          # wherever you cloned it
```

That takes `.venv`, `results/` and everything you wrote with it.

**7. Check it worked**

```bash
ollama list                          # no agentgraph-… entry
echo "$MELLUM_MODEL"                 # empty
```

**Inside WSL2**, two extra things live on the Windows side. Undo the memory setting if you raised
it — edit `%UserProfile%\.wslconfig` in PowerShell, remove the lines you added, then
`wsl --shutdown`:

```ini
[wsl2]
memory=16GB
```

Or remove the whole Ubuntu installation, which takes the model, Ollama and the clone with it in
one command. Only if you installed Ubuntu for this and have nothing else in it:

```powershell
wsl --unregister Ubuntu
```

That is irreversible and deletes everything in that Linux filesystem. Copy out anything you want
to keep first.
</details>

<details>
<summary><b>Windows — native PowerShell</b></summary>

**1. Remove the model** — undoes `ollama pull` + `ollama create`. This is almost all of the disk
space.

```powershell
ollama list                                                             # see what you have
```

```powershell
# Option 1
ollama rm agentgraph-mellum2-thinking hf.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M
```

```powershell
# Option 2
ollama rm qwen3:1.7b
```

Remove the derived model **and** the base model it was built from.

**2. Remove the model variable** — only if you set `MELLUM_MODEL` persistently with `setx` or the
Environment Variables dialog.

```powershell
reg delete HKCU\Environment /F /V MELLUM_MODEL
```

Use `reg delete`, not `setx MELLUM_MODEL ""` — `setx` with an empty value leaves the variable
*present but empty*, which reads as a model named `""` and fails in a much more confusing way than
a missing variable does.

By hand instead: **Settings** → **System** → **About** → **Advanced system settings** →
**Environment variables**, then delete it from the top (user) list.

Either way, already-running programs keep the old value until they restart. For the PowerShell
window you are in:

```powershell
Remove-Item Env:\MELLUM_MODEL -ErrorAction Ignore
```

**3. Remove Ollama itself** — optional, and only if you installed it for this workshop. Do the
model first: the uninstaller does not take the model store with it.

**Settings** → **Apps** → **Installed apps** → **Ollama** → **Uninstall**, or from a terminal if
you installed it with winget:

```powershell
winget uninstall -e --id Ollama.Ollama
```

Then the directories it leaves behind:

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.ollama"
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Ollama"
```

**4. Remove the Docker sandbox image** — only if you built it.

```powershell
docker rmi agentgraph-sandbox
```

**5. Remove `uv` and its cache** — only if you have no other use for it.

```powershell
uv cache clean
uv python uninstall 3.12                                       # only a uv-installed one
Remove-Item -Recurse -Force "$env:USERPROFILE\.local\bin\uv.exe", "$env:LOCALAPPDATA\uv"
```

**6. Delete the clone**

```powershell
Remove-Item -Recurse -Force $HOME\agentfix-react     # wherever you cloned it
```

That takes `.venv`, `results/` and everything you wrote with it.

**7. Check it worked**

```powershell
ollama list                          # no agentgraph-… entry
echo "$env:MELLUM_MODEL"             # empty
```
</details>

### Afterwards

`uv run agentgraph doctor` will fail once you have done any of this, which is the point — you removed
what it checks for. Everything you built still works, and the exercise test still passes, because
it runs against a scripted fake model and never needed a model at all.
