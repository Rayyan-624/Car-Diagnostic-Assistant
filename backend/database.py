"""
Database models for Smart Car Diagnostic Assistant
===================================================
Provides SQLite persistence for diagnosis history using Flask-SQLAlchemy.
Stores each diagnosis session so cases can be reviewed, exported, and
fed back into the Bayesian parameter learning pipeline.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()


class DiagnosisCase(db.Model):
    """
    Stores a single diagnostic session.

    Fields:
        id              Auto-increment primary key.
        timestamp       UTC time of diagnosis.
        symptoms        JSON array of symptom IDs that were observed.
        probabilities   JSON object of {fault_id: probability} posterior.
        top_fault       ID of the most probable fault.
        top_probability Probability of the top fault (for quick sorting).
        top_severity    Severity level of the top fault.
        confirmed_fault Optional: actual fault confirmed by mechanic (for learning).
        notes           Optional free-text notes.
    """
    __tablename__ = 'diagnosis_cases'

    id              = db.Column(db.Integer, primary_key=True)
    timestamp       = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    symptoms        = db.Column(db.Text, nullable=False)       # JSON list of symptom IDs
    probabilities   = db.Column(db.Text, nullable=False)       # JSON dict of probs
    top_fault       = db.Column(db.String(100), nullable=False)
    top_probability = db.Column(db.Float, nullable=False)
    top_severity    = db.Column(db.String(20), nullable=False)
    confirmed_fault = db.Column(db.String(100), nullable=True)  # filled in by mechanic
    notes           = db.Column(db.Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            'id':              self.id,
            'timestamp':       self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'symptoms':        json.loads(self.symptoms),
            'probabilities':   json.loads(self.probabilities),
            'top_fault':       self.top_fault,
            'top_probability': round(self.top_probability, 4),
            'top_severity':    self.top_severity,
            'confirmed_fault': self.confirmed_fault,
            'notes':           self.notes,
        }

    def __repr__(self):
        return (f'<DiagnosisCase id={self.id} top={self.top_fault} '
                f'p={self.top_probability:.2f} at={self.timestamp}>')


def save_case(
    symptoms: dict,
    probabilities: dict,
    top_fault: str,
    top_probability: float,
    top_severity: str,
    confirmed_fault: str = None,
    notes: str = None,
) -> DiagnosisCase:
    """
    Save a diagnosis result to the database.

    Args:
        symptoms:        {symptom_id: 1|0} observed symptoms dict.
        probabilities:   {fault_id: float} posterior probabilities.
        top_fault:       ID of highest-probability fault.
        top_probability: Probability of that fault.
        top_severity:    Severity level string.
        confirmed_fault: Optional actual fault confirmed post-repair.
        notes:           Optional free-text mechanic notes.

    Returns:
        The saved DiagnosisCase instance.
    """
    # Only store symptom IDs that were actually observed (value=1)
    observed = [k for k, v in symptoms.items() if v == 1]

    case = DiagnosisCase(
        symptoms        = json.dumps(observed),
        probabilities   = json.dumps({k: round(v, 4) for k, v in probabilities.items()}),
        top_fault       = top_fault,
        top_probability = top_probability,
        top_severity    = top_severity,
        confirmed_fault = confirmed_fault,
        notes           = notes,
    )
    db.session.add(case)
    db.session.commit()
    return case


def get_all_cases(limit: int = 50, offset: int = 0) -> list:
    """Return recent diagnosis cases, newest first."""
    cases = (DiagnosisCase.query
             .order_by(DiagnosisCase.timestamp.desc())
             .limit(limit)
             .offset(offset)
             .all())
    return [c.to_dict() for c in cases]


def get_cases_as_dataframe():
    """
    Return all confirmed cases as a pandas DataFrame for model retraining.
    Only includes cases where confirmed_fault has been set.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    confirmed = DiagnosisCase.query.filter(
        DiagnosisCase.confirmed_fault.isnot(None)
    ).all()

    if not confirmed:
        return None

    rows = []
    for c in confirmed:
        row = {}
        # Add symptom columns
        observed_symptoms = set(json.loads(c.symptoms))
        for s_id in [
            'engine_not_starting', 'battery_warning_light', 'clicking_sound',
            'dim_headlights', 'warning_lights_all', 'engine_stalling', 'rough_idle',
            'check_engine_light', 'white_smoke', 'excessive_vibration', 'engine_misfire'
        ]:
            row[s_id] = 1 if s_id in observed_symptoms else 0
        # Add fault columns (0 for all, 1 for confirmed)
        for f_id in [
            'battery_dead', 'starter_motor_issue', 'fuel_pump_failure',
            'ignition_coil_issue', 'spark_plugs_fouled', 'head_gasket_failure'
        ]:
            row[f_id] = 1 if f_id == c.confirmed_fault else 0
        rows.append(row)

    return pd.DataFrame(rows) if rows else None