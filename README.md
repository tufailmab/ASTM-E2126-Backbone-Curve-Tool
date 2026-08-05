# ASTM-E2126-Backbone-Curve-Tool

**ASTM-E2126-Backbone-Curve-Tool** is a Python-based graphical application for processing cyclic hysteresis test data and automatically generating backbone (envelope) curves, ASTM E2126 Equivalent Energy Elastic-Plastic (EEEP) bilinear idealizations, stiffness degradation curves, and energy dissipation plots. The application processes experimental displacement-force data stored in text files and produces engineering plots, CSV files, and summary reports.

---

## Project Overview

Experimental cyclic loading tests are commonly performed to evaluate the seismic performance of structural systems. Processing these datasets manually to extract backbone curves and engineering properties is time-consuming and prone to errors.

ASTM-E2126-Backbone-Curve-Tool automates the complete post-processing workflow by implementing the ASTM E2126 EEEP method. The software extracts backbone curves, calculates engineering parameters, generates publication-quality plots, and exports numerical results for further analysis.

The application provides an interactive graphical user interface (GUI) designed for researchers and practicing structural engineers.

---

## Features

- **Backbone Curve Extraction**
  - Automatically extracts positive and negative backbone (envelope) curves from cyclic hysteresis data.

- **ASTM E2126 Bilinear Idealization**
  - Computes the Equivalent Energy Elastic-Plastic (EEEP) bilinear model using ASTM E2126 methodology.

- **Positive, Negative and Average Backbone Curves**
  - Separates positive and negative branches and computes the average backbone response.

- **Engineering Properties**
  - Peak force
  - Peak displacement
  - Yield force
  - Yield displacement
  - Ultimate force
  - Ultimate displacement
  - Initial stiffness
  - Energy dissipation
  - Ductility

- **Cycle Analysis**
  - Automatic cycle identification
  - Stiffness degradation
  - Energy dissipation per loading cycle
  - Cumulative energy dissipation

- **Interactive GUI**
  - Modern Tkinter interface
  - Interactive Matplotlib plotting
  - Zoom and pan
  - Crosshair cursor
  - Live coordinate display
  - Click-to-inspect data points
  - Specimen search
  - Previous/Next specimen navigation

- **Professional Plot Export**
  - PNG
  - PDF
  - SVG
  - Adjustable export DPI
  - Copy plot directly to clipboard

- **Batch Processing**
  - Processes multiple TXT files automatically.

- **Automatic Report Generation**
  - Summary CSV containing engineering properties for every specimen.

---

## Workflow

The application follows the workflow below:

1. Prepare cyclic hysteresis data in TXT format.
2. Select the input folder containing all specimens.
3. Choose an output folder.
4. Process all files with a single click.
5. Review generated plots interactively.
6. Export plots and numerical results.
7. Use the generated CSV files for further structural analysis or publication.

---

## Output Files

The software automatically creates organized folders containing:

- All Backbone Curves (Plots)
- All Backbone Curves (CSVs)
- Positive-Negative-Average Plots
- Positive-Negative-Average CSVs
- Bilinear Idealization Plots
- Bilinear Idealization CSVs
- Stiffness Degradation Plots
- Stiffness Degradation CSVs
- Energy Dissipation Plots
- Energy Dissipation CSVs
- Summary Report

---

## File Information

| Item | Description |
|------|-------------|
| **Script Name** | BBCurve.py |
| **Repository** | ASTM-E2126-Backbone-Curve-Tool |
| **Language** | Python |
| **Input Format** | TXT |
| **Output Formats** | CSV, PNG, PDF, SVG |
| **GUI Framework** | Tkinter |
| **Plotting Library** | Matplotlib |

---

## Usage Instructions

1. Export cyclic hysteresis data as **TXT** files.

2. Place all TXT files inside a single input folder.

3. Run the application:

```bash
python BBCurve.py
```

4. Select:

- Input Folder
- Output Folder

5. Click:

```
Process All TXT Files
```

6. Review results using the interactive plot viewer.

7. Export figures and reports if required.

---

## Requirements

The following software is required:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Tkinter (included with standard Python)

Install the required Python packages using:

```bash
pip install numpy pandas matplotlib
```

---

## Engineering Method

The software implements engineering procedures based on:

- ASTM E2126
- Equivalent Energy Elastic-Plastic (EEEP) Bilinear Idealization
- Backbone (Envelope) Curve Extraction
- Stiffness Degradation Analysis
- Energy Dissipation Analysis
- Ductility Calculation

---

## Intended Users

This software is intended for:

- Structural Engineers
- Earthquake Engineers
- Researchers
- Graduate Students
- Laboratory Engineers
- Universities and Research Institutes

---

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this software under the terms of the license.

---

## Developer Information

- **Developer:** Tufail Mabood
- **Contact:** [WhatsApp](https://wa.me/+923440907874)
- **Note:** This project is open-source. Contributions and improvements are welcome.

This project is open-source. Contributions, feature requests, and improvements are welcome.
