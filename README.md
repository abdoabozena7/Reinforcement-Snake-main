# Reinforcement Learning Snake

## Overview

This project is an implementation of a Deep Q-Network (DQN) agent that learns to play the classic Snake game through reinforcement learning. The agent interacts with a custom game environment, improves its behavior over time, and learns optimal actions based on rewards and penalties.

The entire system runs in the browser using TensorFlow.js and includes real-time training visualization.

## How It Works

The agent observes the game state, which is represented by a set of numerical features describing the position of the snake, food, and nearby hazards. Based on this state, the neural network estimates the expected future reward for each possible action.

The learning process is based on the Q-learning principle, where the model updates its predictions using the Bellman equation and improves through repeated gameplay experience.

## Key Components

Environment
A grid-based Snake game where the agent controls movement, collects food, and avoids collisions.

Deep Q-Network
A neural network that maps game states to action values and guides decision-making.

Experience Replay
Past experiences are stored and randomly sampled during training to stabilize learning.

Epsilon-Greedy Strategy
The agent balances exploration and exploitation by gradually reducing random actions over time.

## Features

Real-time training in the browser
Live performance visualization
Persistent model storage using browser memory
Training progress tracking with score history

## Training Process

The agent starts with completely random behavior and gradually improves through repeated episodes. Over time, it learns to maximize score by reaching food efficiently while avoiding collisions.

## Technologies Used

TensorFlow.js
JavaScript
HTML Canvas
Chart Visualization Library

## Goal

The goal of this project is to demonstrate how reinforcement learning can be applied to a simple environment and how an agent can learn complex behavior through trial and error without explicit programming of rules.
