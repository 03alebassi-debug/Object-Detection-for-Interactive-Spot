# Interactive Spot: Object Recognition Module

This repository houses the **Object Recognition** subsystem for the broader **Interactive Spot** project. 

Interactive Spot is an autonomous robotics system designed to navigate physical environments, process natural voice inputs, and execute basic physical operations such as "find" or "pick." This specific module is responsible for the robot's visual perception and target identification capabilities. 

> **Note:** This repository is currently being developed as a standalone module. Once the features below are fully implemented and tested, this codebase will be merged into the official, primary Interactive Spot repository.

## 🎯 Module Overview
The goal of this module is to seamlessly connect user voice commands with visual processing. By leveraging open-vocabulary object recognition models, this system dynamically interprets what the user wants the robot to interact with (e.g., "a ball") and identifies that object in the real world, computing its physical location for the robot to act upon.

## 🗺️ Development Roadmap & To-Do List

- [x] **Establish Baseline Detection:** Integrate and verify NanoOWL to ensure the core open-vocabulary object recognition is functioning correctly.
- [x] **Dynamic Prompting:** Implement multithreading to allow the system to continuously update and process search prompts in real-time without blocking the main execution loop.
- [x] **Spatial Awareness:** Extract and process depth information to determine the physical distance and coordinates of the recognized objects.
- [ ] **Command Parsing & Integration:** Develop a parser to process multi-modal commands (e.g., *"pick a ball"*). This step will:
  1. Extract the target object (*"ball"*) and inject it into the object recognition prompt.
  2. Extract the action (*"pick"*) and trigger the corresponding physical routine in the robot's main control system.