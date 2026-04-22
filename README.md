# Reinforcement Learning Snake

## Overview

This version separates the project into:

- `FastAPI` back-end for the snake environment, replay buffer, and DQN training loop
- Static `HTML/CSS/JavaScript` front-end for rendering the board, chart, and dashboard

The browser no longer owns the reinforcement logic. It only calls the API and redraws the state returned by Python.

## Project Structure

- `main.py`: FastAPI entrypoint
- `backend/environment.py`: snake game environment
- `backend/agent.py`: replay buffer and NumPy DQN agent
- `backend/session.py`: training session state and persistence
- `static/index.html`: front-end markup
- `static/styles.css`: front-end styles
- `static/app.js`: rendering and API polling

## Run

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the server:

```powershell
uvicorn main:app --reload
```

4. Open:

```text
http://127.0.0.1:8000
```

## Notes

- Model weights are saved to `storage/model_weights.npz`
- Training metadata is saved to `storage/training_state.json`
- The front-end polls `/api/tick` to advance one training frame at a time
