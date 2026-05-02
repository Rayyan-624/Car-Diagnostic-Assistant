# Smart Car Diagnostic Assistant

An intelligent, AI-powered car diagnostic system built with a **Bayesian Network** that analyzes vehicle symptoms to identify probable faults and provide actionable repair recommendations.

## 🎯 Overview

The Smart Car Diagnostic Assistant uses probabilistic reasoning to:
- **Analyze symptoms** selected by the user
- **Calculate fault probabilities** based on a trained Bayesian network
- **Rank faults** by likelihood and severity
- **Recommend diagnostic steps** through differential diagnosis
- **Learn from confirmed cases** to continuously improve accuracy

## 🏗️ Architecture

### Backend (Flask API)
- **Framework**: Flask with Flask-CORS for cross-origin requests
- **AI Engine**: Bayesian Network via `pgmpy` for probabilistic inference
- **Database**: SQLite for diagnosis case history and learning
- **RESTful API**: Comprehensive endpoints for diagnosis, symptom retrieval, and case management

### Frontend (Web UI)
- **Framework**: Vanilla HTML/CSS/JavaScript
- **Design**: Modern dark theme with responsive layout
- **Features**: Interactive symptom selection, real-time diagnosis, case history

## 📋 Features

### Core Diagnosis
- **Full Diagnosis**: Complete probability distribution for all faults
- **Quick Diagnosis**: Top-3 most likely faults only
- **Confidence Estimates**: Uncertainty quantification for each diagnosis
- **Differential Diagnosis**: Intelligent symptom suggestions to narrow down faults

### Case Management
- **Case History**: View all past diagnoses with pagination
- **Case Confirmation**: Record actual faults to train the model
- **Model Retraining**: Update network parameters from confirmed cases

### Extensibility
- **Configuration-Driven**: Symptoms and faults defined in `symptoms_config.json`
- **Custom Metadata**: Labels, severity levels, and recommended actions per symptom/fault
- **No Code Changes**: Add or modify diagnostic rules without touching Python code

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Car Diagnostic Assistant"
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   cd backend
   python app.py
   ```
   The API will be available at `http://localhost:5000`

2. **Open the frontend**
   - Open `frontend/index.html` in your web browser
   - Or serve it with a simple HTTP server:
     ```bash
     cd frontend
     python -m http.server 8000
     ```
   - Access at `http://localhost:8000`

## 📡 API Endpoints

All endpoints are prefixed with `/api`.

### Health & Status
- `GET /api/health` - Health check and network status

### Symptoms
- `GET /api/symptoms` - Retrieve all selectable symptoms
- `GET /api/symptom-description/<id>` - Get metadata for a specific symptom

### Diagnosis
- `POST /api/diagnose` - Full diagnosis with complete probability distribution
- `POST /api/diagnose/quick` - Top-3 faults only
- `POST /api/diagnose/confidence` - Diagnosis with uncertainty estimates
- `POST /api/next-symptom` - Get best next symptom to check (differential diagnosis)

### Network & Visualization
- `GET /api/network` - Retrieve Bayesian network structure for visualization

### Case Management
- `GET /api/cases` - Diagnosis history (paginated)
- `POST /api/cases/<id>/confirm` - Record confirmed fault for a case
- `POST /api/train` - Retrain model from confirmed cases

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the `backend` directory:
```env
DATABASE_URI=sqlite:///diagnostics.db
FLASK_ENV=development
```

### Symptoms & Faults Configuration
Edit `backend/symptoms_config.json` to define:
- Available symptoms users can select
- Fault nodes and their descriptions
- Severity levels and recommended actions
- Custom metadata

Example structure:
```json
{
  "symptoms": {
    "check_engine_light": {
      "label": "Check Engine Light",
      "severity": "high",
      "description": "Malfunction indicator lamp is on"
    }
  },
  "faults": {
    "spark_plugs_fouled": {
      "label": "Fouled Spark Plugs",
      "severity": "medium",
      "recommendation": "Replace spark plugs"
    }
  }
}
```

## 📊 Project Structure

```
Car Diagnostic Assistant/
├── backend/
│   ├── app.py                    # Flask API entry point
│   ├── bayesian_network.py       # Bayesian network logic & inference
│   ├── database.py               # SQLite database models
│   ├── symptoms_config.json      # Symptoms & faults configuration
│   ├── requirements.txt          # Python dependencies
│   ├── instance/                 # SQLite database storage
│   └── tests/
│       ├── __init__.py
│       └── test_network.py       # Unit tests
├── frontend/
│   └── index.html                # Web UI
└── README.md                      # This file
```

## 🧪 Testing

Run the test suite:
```bash
cd backend
pytest tests/
```

Or run specific tests:
```bash
pytest tests/test_network.py -v
```

## 📦 Dependencies

### Backend
- **Flask** (2.3.3) - Web framework
- **Flask-CORS** (4.0.0) - Cross-origin request handling
- **Flask-SQLAlchemy** (3.1.1) - Database ORM
- **pgmpy** (0.1.23) - Probabilistic graphical models
- **NetworkX** (3.1) - Graph algorithms
- **NumPy** (<2.0.0) - Numerical computing
- **Pandas** (≥1.5.0) - Data analysis
- **python-dotenv** (1.0.0) - Environment configuration

## ⚙️ Technical Details

### Bayesian Network
- **Nodes**: Faults (hidden) and symptoms (observable)
- **Edges**: Causal relationships between faults and their manifestations
- **Inference Engine**: Variable Elimination for exact probabilistic inference
- **Learning**: Parameter learning from confirmed diagnosis cases

### Database Schema
- **DiagnosisCase**: Stores diagnosis history
  - Selected symptoms
  - Calculated probabilities
  - Confirmed fault (optional)
  - Timestamp

### Key Fixes & Improvements
- ✅ Fixed pgmpy/NumPy compatibility (pgmpy 0.1.23 requires NumPy <2.0.0)
- ✅ Corrected engine_misfire classification (moved from fault to symptom)
- ✅ Improved network structure detection using explicit node set membership
- ✅ Lazy Bayesian network initialization to prevent import-time crashes
- ✅ Metadata-driven configuration from JSON (no code changes needed)
- ✅ Comprehensive error handling and logging

## 🎓 Use Cases

1. **Auto Repair Shops**: Assist mechanics in fault diagnosis
2. **Vehicle Owners**: Quick self-diagnosis before visiting repair shop
3. **Training**: Educational tool for understanding vehicle diagnostics
4. **Research**: Baseline for Bayesian inference applications

## 🐛 Troubleshooting

### API won't start
- Ensure Python 3.8+ is installed
- Check that all dependencies are installed: `pip install -r backend/requirements.txt`
- Verify port 5000 is not in use

### Diagnosis returns 503 error
- Check backend logs for pgmpy/NumPy compatibility issues
- Verify NumPy version: `pip list | grep -i numpy`
- Should be <2.0.0 for pgmpy 0.1.23 compatibility

### Database errors
- Delete `backend/instance/diagnostics.db` to reset
- Ensure write permissions in the `backend` directory

## 📈 Future Enhancements

- [ ] Web-based Bayesian network visualization
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Advanced statistics dashboard for repair shops
- [ ] Integration with vehicle repair manuals
- [ ] Predictive maintenance recommendations

## 📝 License

This project is created as part of AI Lab coursework.

## 👥 Author

Created as part of Semester 6 AI Lab project at FAST-NUCES

## 📞 Support

For issues, questions, or suggestions, please check the project repository or contact the development team.

---

**Last Updated**: May 2, 2026
