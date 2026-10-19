# wuji-mujoco-sim

Minimal simulation demo for **hand1** and **hand2** in **MuJoCo 3.11.0**. Loads the hand model and runs a wave motion in a loop. Both left and right hands are supported.

Based on [mujoco-sim](https://github.com/wuji-technology/mujoco-sim).

## Simulation preview

<p align="center">All videos below are rendered from MuJoCo simulation. No real-robot footage is used.</p>

<table align="center">
  <thead>
    <tr>
      <th align="center">hand1</th>
      <th align="center">hand2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="50%" align="center"><video src="assets/hand1-simulation.mp4" width="100%" controls muted loop></video></td>
      <td width="50%" align="center"><video src="assets/hand2-simulation.mp4" width="100%" controls muted loop></video></td>
    </tr>
  </tbody>
</table>

## Quick Start

### Installation

Requires Python 3.10 or newer.

```bash
git clone --recurse-submodules https://github.com/wuji-technology/wuji-mujoco-sim.git
cd wuji-mujoco-sim
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### Running

```bash
# hand1, right hand (default)
.venv/bin/python run_sim.py --hand hand1 --side right

# hand1, left hand
.venv/bin/python run_sim.py --hand hand1 --side left

# hand2, right hand
.venv/bin/python run_sim.py --hand hand2 --side right

# hand2, left hand
.venv/bin/python run_sim.py --hand hand2 --side left
```

Close the simulation window to stop.

### Update Models

```bash
git submodule update --init --recursive --remote
```
