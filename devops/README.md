# Metta AI Devops

Scripts for setting up a Metta AI development environment and launching cloud jobs and sandboxes.

Ensure you've run through installation as described in the primary README.

## Launching Sandbox Environments

To launch a new sandbox on AWS:

```bash
./devops/skypilot/sandbox.py
```

## Launching Train Jobs

See [skypilot README](./skypilot/README.md).

<br></br>

## Using Remote Machines in Cursor IDE

Follow these steps to connect to remote machines directly through Cursor's interface:

### Prerequisites

- SkyPilot must be configured and running
- Remote machines should be accessible via SSH

### Setup Steps

1. **Install the Remote SSH Extension**

   - Open Cursor Extensions (Ctrl+Shift+X or Cmd+Shift+X)
   - Search for and install "Remote - SSH"

2. **Access Remote Explorer**

   - Look for the "Remote Explorer" tab in Cursor's sidebar
   - This will appear after installing the Remote SSH extension

3. **Connect to Your Machine**

   - In the Remote Explorer panel, find your machine listed under "SSH TARGETS"
   - Your SkyPilot-managed machines should appear automatically
   - Right-click on your target machine and select "Connect to Host in New Window"

4. **Open Your Project**
   - A new Cursor window will open connected to the remote machine
   - Use "Open Folder" to navigate to your project directory
   - You can now edit files, run terminals, and debug directly on the remote machine

### Benefits

- Full IDE experience on remote hardware
- Seamless file editing and debugging
- Integrated terminal access
- Extension support on remote machines
