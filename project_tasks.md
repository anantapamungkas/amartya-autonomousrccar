## 📘 Project Title: Camera Calibration & Color Segmentation Tool

### 🧩 Project Modules
1. RealSense Pipeline
2. HSV/YUV Color Calibration
3. Masking & Contour Detection
4. Depth Estimation
5. GUI Trackbar Tuning
6. Utilities Packaging (reusable modules)

---

### 🔄 Task Board

| Task ID | Task Description                         | Assigned To | Status        | Deadline   | Notes                                     |
|---------|-------------------------------------------|-------------|---------------|------------|-------------------------------------------|
| T01     | Refactor calibration code into `utils.py` | Ananta      | ✅ Done        | 13 Jun 25  |                                           |
| T02     | Create `main.py` using the new utils      | Team Member | ⏳ In Progress | 14 Jun 25  | Use new `start_calibration()` function    |
| T03     | Add functionality to save/load HSV values | TBD         | ⏳ Planned     | 15 Jun 25  | Store in JSON                             |
| T04     | Add GUI button to start/stop calibration  | TBD         | 🔲 Not Started | 16 Jun 25  | Optional: Use tkinter or PyQt             |
| T05     | Integrate object tracking (e.g. centroid) | TBD         | 🔲 Not Started | 17 Jun 25  | Use OpenCV tracker                        |
| T06     | Write README with usage instructions      | TBD         | 🔲 Not Started | 17 Jun 25  | Add example usage for devs                |

---

### 🧭 Project Directory Structure

```
camera_calibration_project/
│
├── main.py                  # Runs the main calibration loop
├── utils/
│   └── calibration_utils.py # Reusable functions (trackbars, processing)
├── saved_values/
│   └── hsv_config.json      # Optional: Load/save HSV ranges
├── README.md
├── project_tasks.md         # ← This File
```

---

### 🗓️ Weekly Goals

| Week | Goal                                     | Owner   |
|------|------------------------------------------|---------|
| W1   | Refactor & modularize code               | Ananta  |
| W2   | Add load/save HSV, test integration      | Team    |
| W3   | Document & prepare for integration tests | Team    |

