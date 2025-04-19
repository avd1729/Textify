# Textify 

**Textify** is an intelligent, privacy-first Android keyboard app that provides next-word suggestions through a lightweight, locally embedded **n-gram prediction model**. Designed with modern UI principles using **Jetpack Compose** and powered by a **federated learning backend**, Textify adapts to users' writing styles while keeping their data secure on-device.

---
![image](https://github.com/user-attachments/assets/ad3c6d3e-ba3f-4d55-80ba-b53afe6d61c3)
---
## What Makes Textify Special?

- **On-device Prediction**:  
  Textify embeds a Python-based n-gram model within the Android application itself, enabling real-time next-word suggestions without requiring a network connection or external inference engine.

- **Federated Learning for Privacy**:  
  As users type, the model learns and adapts locally. Instead of uploading raw input data, the keyboard only shares **model weight updates** with a centralized server. This preserves user privacy while enabling the model to improve globally.

- **Global Aggregation via Cloud**:  
  A backend service, written in Python and deployed on **AWS EC2**, receives these model updates from various devices. It aggregates them to form a refined global model, which is periodically redistributed to all clients, completing the federated learning cycle.

---

## Core Components

- **Android App (Kotlin + Jetpack Compose)**:  
  A sleek and responsive keyboard UI that captures user input, runs local prediction using embedded Python, and manages communication with the global model server.

- **Embedded Python Engine**:  
  A minimal n-gram model written in Python is integrated into the Android app using  Chaquopy. It handles local training and prediction, entirely on-device.

- **Federated Update Flow**:  
  After local training on user data:
  - The app serializes the updated model weights.
  - Sends them to the cloud server (AWS EC2).
  - The server aggregates multiple updates to refine the global model.
  - Updated models are sent back to the clients for continual improvement.

- **Backend Server (Python/Flask)**:  
  A simple and scalable API deployed on EC2 that:
  - Accepts model updates from multiple clients.
  - Performs aggregation (e.g., averaging weights).
  - Returns the improved model for future use.

---

##  Privacy by Design

Textify is built with privacy at its core:
- No user-typed content ever leaves the device.
- Only abstracted model updates are sent.
- Enables global improvement without compromising individual user data.

---

## Vision & Future Scope

Textify aims to combine usability, intelligence, and privacy in a seamless typing experience. Some of the future directions include:

- Expanding from n-gram to neural next-word models (while maintaining performance and privacy).
- Multilingual support for localized predictions.
- Integration of differential privacy during update sharing.
- Real-time insights into local vs global model improvements.
- Periodic background syncs to minimize battery and bandwidth impact.

---

## Summary

**Textify** is not just a keyboard — it’s a smarter, privacy-conscious writing assistant. By blending lightweight NLP techniques with federated learning and modern Android design, it brings intelligent text prediction to mobile devices without ever sacrificing user trust.

