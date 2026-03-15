# K1 Walking & Talking

Voice-controlled robot system for the Booster K1 — talk to your robot and it walks, follows people, tracks objects, dances, waves, and more. Powered by Google Gemini Live API + YOLOv8 + stereo depth + face recognition.

## Connecting to the Robot

### Network Setup

The K1 robot communicates over Ethernet. Your control machine (Jetson, laptop, etc.) must be on the same network as the robot.

1. **Wired connection (recommended):** Connect an Ethernet cable between your machine and the robot's Ethernet port.

2. **Find your network interface name:**
   ```bash
   ip addr
   # Look for an interface like eth0, eth1, eno1, or enp0s31f6
   # The interface connected to the robot will typically have a 192.168.x.x address
   ```

3. **Verify connectivity:**
   ```bash
   # The robot's default IP is usually on the same subnet
   ping 192.168.1.120   # adjust to your robot's IP
   ```

4. **If running directly on the robot's onboard Jetson**, use the loopback or the internal interface:
   ```bash
   python3 gemini_robot_control.py eth0
   # or if running locally on the robot:
   python3 gemini_robot_control.py 127.0.0.1
   ```

### ROS 2 Camera Bridge

The robot's stereo cameras publish frames via ROS 2 topics. Ensure the camera bridge node is running:

```bash
# Check that camera topics are active
ros2 topic list | grep booster_camera_bridge

# Expected topics:
#   /booster_camera_bridge/image_left_raw
#   /booster_camera_bridge/image_right_raw
```

If the topics are not available, start the camera bridge:
```bash
ros2 launch booster_camera_bridge camera_bridge.launch.py
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/arminforoughi/k1_walking_talking.git
cd k1_walking_talking

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
export GEMINI_API_KEY="your-key-here"

# 4. Run (replace eth0 with your network interface)
python3 gemini_robot_control.py eth0

# 5. Open the web UI in your browser
#    http://<robot-ip>:8080
```

## Overview

| Script | Description |
|--------|-------------|
| **server.py** | Remote server — runs YOLO + face + Gemini on your PC (recommended) |
| **robot_client.py** | Thin client — runs on the robot, streams video, executes commands |
| **gemini_robot_control.py** | All-in-one on-robot mode (original, slower) |
| **gemini_live_camera.py** | Gemini Live camera stream (no robot control) |

## Remote Mode (Recommended)

Offload all heavy computation (YOLO, face recognition, Gemini AI) to your PC/laptop. The robot runs a thin client that streams camera frames and executes commands. Much faster and more responsive.

### Architecture

```
┌──────────────────────┐       WebSocket (ws://IP:9090)       ┌──────────────────────┐
│   ROBOT (K1)         │ ◄═══════════════════════════════════► │   YOUR MACHINE       │
│   robot_client.py    │                                       │   server.py          │
│                      │  Robot → Server:                      │                      │
│  • ROS2 camera sub   │  • JPEG video frames (10 fps)        │  • YOLO detection    │
│  • ROS2 depth sub    │  • Depth maps (compressed)           │  • Face recognition  │
│  • Robot SDK control │  • Audio from robot mic              │  • Gemini Live API   │
│  • Dance choreography│                                       │  • Tracking/Follow   │
│  • Audio playback    │  Server → Robot:                      │  • Web UI (:8080)    │
│                      │  • Move/head/dance commands (JSON)   │  • Audio I/O         │
│                      │  • Gemini speech audio               │                      │
└──────────────────────┘                                       └──────────────────────┘
```

### Quick Start (Remote Mode)

**1. On your PC/laptop (the server):**

```bash
pip install websockets

export GEMINI_API_KEY="your-key-here"
python3 server.py --voice Puck
# Web UI at http://localhost:8080
# Robot WebSocket at ws://0.0.0.0:9090
```

**2. On the robot:**

```bash
pip install websockets

python3 robot_client.py eth0 --server ws://YOUR_PC_IP:9090
```

### Server Arguments

```
python3 server.py [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--api-key` | `$GEMINI_API_KEY` | Gemini API key |
| `--voice` | `Puck` | Voice: Puck, Charon, Kore, Fenrir, Aoede |
| `--port` | `8080` | Web UI port |
| `--ws-port` | `9090` | WebSocket port for robot connection |
| `--model` | `yolov8n.pt` | YOLO model path |
| `--confidence` | `0.5` | Detection confidence threshold |
| `--no-faces` | off | Disable face recognition |
| `--follow-distance` | `1.0` | Target follow distance in meters |
| `--audio-source` | `robot` | `robot` = stream from robot mic, `local` = use PC mic |
| `--mic-gain` | `3.0` | Mic gain (for `--audio-source local`) |

### Robot Client Arguments

```
python3 robot_client.py <interface> [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `interface` | *(required)* | Network interface for robot SDK (e.g. `eth0`) |
| `--server` | `ws://localhost:9090` | Server WebSocket URL |
| `--fps` | `10` | Video stream FPS |
| `--depth-fps` | `5` | Depth stream FPS |
| `--mic-gain` | `3.0` | Robot mic gain multiplier |
| `--mic-device` | auto | PyAudio mic device index |

## On-Robot Mode (`gemini_robot_control.py`)

The original all-in-one script — runs everything on the robot. Simpler setup but slower since the robot handles all ML inference.

### Features

- **Voice conversation** via Gemini Live API (bidirectional audio streaming)
- **Real-time vision** — camera frames sent to Gemini for scene understanding
- **Person following** with obstacle avoidance (stereo depth + 5-zone scanning)
- **Object tracking** — head follows any detected object class by name
- **Go-to navigation** — walk toward a detected object (e.g. "go to the chair")
- **Face recognition** with persistent name caching (learns names via voice)
- **Dance library** — 15+ dances including moonwalk, salsa, macarena, dab, etc.
- **Movement commands** — walk forward/backward, turn, strafe, approach, back up
- **Head control** — look left/right/up/down/center, nod, shake head
- **Hand gestures** — wave, handshake, flex
- **Web UI** at port 8080 with live detection feed, chat log, and text input

### Voice Commands (Understood by Gemini)

The robot understands natural language. Example commands:

| Category | Examples |
|----------|----------|
| **Follow** | "Follow me", "Follow that person", "Follow John" |
| **Stop** | "Stop following", "Stop" |
| **Go to** | "Go to the chair", "Walk to the table" |
| **Track** | "Look at the dog", "Track that person" |
| **Dance** | "Dance", "Do the moonwalk", "Salsa", "Macarena" |
| **Wave** | "Wave hello" |
| **Move** | "Walk forward", "Turn left", "Back up", "Strafe right" |
| **Look** | "Look left", "Look up", "Look at me" |
| **Gestures** | "Handshake", "Dab", "Flex", "Bow" |
| **Face** | "My name is John" (learns your face) |

### Command-Line Arguments

```
python3 gemini_robot_control.py <interface> [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `interface` | *(required)* | Network interface for robot SDK (e.g. `eth0`, `127.0.0.1`) |
| `--api-key` | `$GEMINI_API_KEY` | Gemini API key |
| `--voice` | `Puck` | Voice: Puck, Charon, Kore, Fenrir, Aoede |
| `--frame-interval` | `1.0` | Seconds between frames sent to Gemini |
| `--port` | `8080` | Web UI port |
| `--model` | `yolov8n.pt` | YOLO model path |
| `--confidence` | `0.5` | Detection confidence threshold |
| `--face-tolerance` | `0.6` | Face recognition tolerance (lower = stricter) |
| `--no-faces` | off | Disable face recognition |
| `--follow-distance` | `1.0` | Target follow distance in meters |
| `--mic-gain` | `3.0` | Software mic gain multiplier |
| `--mic-device` | auto | PyAudio input device index |

### Web UI

Open `http://<robot-ip>:8080` in your browser:

- Live camera feed with YOLO bounding boxes, face labels, and depth info
- Detection sidebar listing all visible objects with distances
- Chat log showing the full Gemini conversation
- Text input field for typing commands (in addition to voice)

## K1 Person Following (`k1_follow_person.py`)

Standalone person following without Gemini. Uses YOLO + MiDaS depth estimation.

```bash
python3 k1_follow_person.py eth0 --yolo yolov8n.pt --depth small --web --port 8080 --auto-follow
```

- Head tracking mode (default) or full follow mode
- Web interface with follow toggle and person selection
- Configurable depth model: small (fast), hybrid, large (accurate)

## Requirements

### System Requirements

- **Booster K1 robot** with Booster Robotics SDK
- **ROS 2** (Humble or later) with camera bridge running
- **Python 3.8+**
- **CUDA GPU** (optional but recommended for real-time inference)
- **Microphone + Speaker** for voice interaction (iFlytek mic auto-detected)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `websockets` — robot ↔ server communication (needed on both sides)
- `booster-robotics-sdk-python` — K1 robot SDK (robot only)
- `google-genai` — Gemini Live API (server only)
- `ultralytics` — YOLOv8 (server only)
- `face_recognition` — face detection and recognition (server only)
- `opencv-python` — image processing (both)
- `pyaudio` — audio I/O (both)
- `rclpy`, `sensor_msgs`, `cv_bridge` — ROS 2 camera interface (robot only)
- `numpy` — numerical operations (both)

## Troubleshooting

### Robot Connection

**"Cannot connect to robot":**
- Verify the network interface: `ip addr`
- Check you're on the same subnet as the robot
- Try `eth0` for wired or `127.0.0.1` if running on the robot's Jetson

**Robot not moving:**
- The script auto-switches to walking mode on startup
- If the robot is in a fault state, power cycle it and try again
- Check the terminal output for SDK error messages

### Camera

**No camera feed:**
- Verify ROS 2 topics: `ros2 topic list | grep booster_camera_bridge`
- Check the camera bridge is running
- Inspect topic output: `ros2 topic echo /booster_camera_bridge/image_left_raw --once`

### Audio

**Mic not working:**
- List audio devices: `python3 -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"`
- Use `--mic-device <index>` to select manually
- Increase gain with `--mic-gain 5.0` if too quiet

**No audio output from robot:**
- Check speaker is connected and volume is up
- Test: `speaker-test -t wav -c 2`

### Web UI

**Can't access web UI:**
- Check firewall: `sudo ufw allow 8080`
- Verify port: `netstat -tulpn | grep 8080`
- Try a different port: `--port 8081`

## License

MIT License
