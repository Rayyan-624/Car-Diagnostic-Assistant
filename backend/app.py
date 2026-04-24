"""
Flask API — Smart Car Diagnostic Assistant
==========================================
All endpoints are prefixed with /api.

Endpoints:
  GET  /api/health                     Health check + network status
  GET  /api/symptoms                   All selectable symptoms
  GET  /api/network                    Bayesian network structure for visualization
  POST /api/diagnose                   Full diagnosis (probabilities + recs)
  POST /api/diagnose/quick             Top-3 faults only
  POST /api/diagnose/confidence        Diagnosis with uncertainty estimates
  POST /api/next-symptom               Differential Dx: best next symptom to check
  GET  /api/cases                      Diagnosis history (paginated)
  POST /api/cases/<id>/confirm         Record confirmed fault for a case
  POST /api/train                      Retrain model from confirmed cases
  GET  /api/symptom-description/<id>   Single symptom metadata
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Database config ───────────────────────────────────────────────────────────
DB_PATH = os.environ.get('DATABASE_URI', 'sqlite:///diagnostics.db')
app.config['SQLALCHEMY_DATABASE_URI']        = DB_PATH
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from database import db, save_case, get_all_cases, get_cases_as_dataframe, DiagnosisCase
db.init_app(app)

with app.app_context():
    db.create_all()

# ── Lazy Bayesian Network init ────────────────────────────────────────────────
# Module-level crash on import is prevented — if pgmpy/numpy fails, the API
# still starts and returns 503 for diagnosis endpoints with a clear error message.
_network      = None
_network_error = None

def get_network():
    global _network, _network_error
    if _network is not None:
        return _network
    if _network_error is not None:
        return None
    try:
        from bayesian_network import CarDiagnosticNetwork
        _network = CarDiagnosticNetwork()
        logger.info("Bayesian Network loaded successfully.")
    except Exception as e:
        _network_error = str(e)
        logger.error(f"Failed to initialize Bayesian Network: {e}")
    return _network


def require_network(fn):
    """Decorator: returns 503 if the network failed to load."""
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        net = get_network()
        if net is None:
            return jsonify({
                'error':   'AI model is not available.',
                'detail':  _network_error or 'Unknown initialization error.',
                'hint':    'Check that pgmpy==0.1.23 and numpy<2.0 are installed.'
            }), 503
        return fn(*args, **kwargs)
    return wrapper


def _normalize_symptoms(raw) -> dict:
    """
    Accept symptoms in either format:
      - Dict:  {"engine_not_starting": 1, "rough_idle": 1}  ← correct
      - List:  ["engine_not_starting", "rough_idle"]         ← also accepted

    Returns normalized dict {symptom_id: 1}.
    Raises ValueError if input is neither.
    """
    if isinstance(raw, list):
        return {s: 1 for s in raw if isinstance(s, str)}
    if isinstance(raw, dict):
        return {k: int(v) for k, v in raw.items() if isinstance(k, str)}
    raise ValueError(f"'symptoms' must be a list or dict, got {type(raw).__name__}")


# ── Health ────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    net    = get_network()
    status = 'ok' if net else 'degraded'
    return jsonify({
        'status':          status,
        'network_ready':   net is not None,
        'network_error':   _network_error,
        'message':         'Car Diagnostic API is running' if net else 'AI model failed to load.',
    }), 200 if net else 503


# ── Symptoms ──────────────────────────────────────────────────────────────────

@app.route('/api/symptoms', methods=['GET'])
@require_network
def get_symptoms():
    net   = get_network()
    metas = net.get_symptom_labels()
    return jsonify({'symptoms': [
        {'id': sid, **meta} for sid, meta in metas.items()
    ]}), 200


# ── Network structure ─────────────────────────────────────────────────────────

@app.route('/api/network', methods=['GET'])
@require_network
def get_network_structure():
    return jsonify(get_network().get_network_structure()), 200


# ── Diagnose ──────────────────────────────────────────────────────────────────

@app.route('/api/diagnose', methods=['POST'])
@require_network
def diagnose():
    """
    Full diagnosis endpoint.

    Body: {"symptoms": {"engine_not_starting": 1, "rough_idle": 1}}
       or {"symptoms": ["engine_not_starting", "rough_idle"]}

    Returns probabilities, recommendations, observed symptoms, and top fault.
    """
    try:
        data     = request.get_json(force=True) or {}
        symptoms = _normalize_symptoms(data.get('symptoms', {}))

        if not any(v == 1 for v in symptoms.values()):
            return jsonify({'error': 'At least one symptom must have value 1.'}), 400

        net              = get_network()
        fault_probs      = net.diagnose(symptoms)
        recommendations  = net.get_recommendations(fault_probs)

        sorted_faults = sorted(fault_probs.items(), key=lambda x: x[1], reverse=True)
        top_fault, top_prob = sorted_faults[0]
        top_rec = next((r for r in recommendations if r['fault'] == top_fault), {})

        # Persist to database
        save_case(
            symptoms        = symptoms,
            probabilities   = fault_probs,
            top_fault       = top_fault,
            top_probability = top_prob,
            top_severity    = top_rec.get('severity', 'medium'),
        )

        return jsonify({
            'success':           True,
            'probabilities':     fault_probs,
            'recommendations':   recommendations,
            'observed_symptoms': symptoms,
            'top_fault':         top_fault,
            'top_probability':   round(top_prob, 4),
        }), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Diagnosis failed")
        return jsonify({'error': 'Diagnosis error.', 'detail': str(e)}), 500


@app.route('/api/diagnose/quick', methods=['POST'])
@require_network
def diagnose_quick():
    """Return only the top-3 probable faults."""
    try:
        data     = request.get_json(force=True) or {}
        symptoms = _normalize_symptoms(data.get('symptoms', {}))
        net      = get_network()
        probs    = net.diagnose(symptoms) if any(v == 1 for v in symptoms.values()) \
                   else net._get_prior_probabilities()
        return jsonify({
            'success':    True,
            'top_faults': net.get_recommendations(probs)[:3],
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Quick diagnosis failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/diagnose/confidence', methods=['POST'])
@require_network
def diagnose_with_confidence():
    """
    Diagnosis with uncertainty quantification (±confidence intervals).
    Slower than /diagnose — runs 20 inference perturbations.
    """
    try:
        data        = request.get_json(force=True) or {}
        symptoms    = _normalize_symptoms(data.get('symptoms', {}))
        n_perturb   = min(int(data.get('n_perturbations', 20)), 50)
        net         = get_network()
        confidence  = net.diagnose_with_confidence(symptoms, n_perturbations=n_perturb)
        return jsonify({'success': True, 'confidence': confidence}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Confidence diagnosis failed")
        return jsonify({'error': str(e)}), 500


# ── Differential diagnosis ────────────────────────────────────────────────────

@app.route('/api/next-symptom', methods=['POST'])
@require_network
def next_best_symptom():
    """
    Differential diagnosis: given current evidence and the top suspected fault,
    return the symptom whose observation would maximally change the diagnosis.

    Body: {
        "symptoms":  {"engine_not_starting": 1},
        "top_fault": "battery_dead"
    }
    """
    try:
        data       = request.get_json(force=True) or {}
        symptoms   = _normalize_symptoms(data.get('symptoms', {}))
        top_fault  = data.get('top_fault', '')
        net        = get_network()

        if not top_fault:
            probs     = net.diagnose(symptoms)
            sorted_p  = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top_fault = sorted_p[0][0] if sorted_p else ''

        symptom_id, gain = net.get_next_best_symptom(symptoms, top_fault)
        symptom_meta     = net.get_symptom_labels().get(symptom_id or '', {})

        return jsonify({
            'success':           True,
            'next_symptom_id':   symptom_id,
            'next_symptom_label': symptom_meta.get('label', symptom_id),
            'information_gain':  gain,
            'for_fault':         top_fault,
        }), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Next-symptom query failed")
        return jsonify({'error': str(e)}), 500


# ── Case history ──────────────────────────────────────────────────────────────

@app.route('/api/cases', methods=['GET'])
def get_cases():
    """Return diagnosis history, newest first. Supports ?limit= and ?offset=."""
    try:
        limit  = min(int(request.args.get('limit',  20)), 100)
        offset = max(int(request.args.get('offset',  0)),   0)
        cases  = get_all_cases(limit=limit, offset=offset)
        total  = DiagnosisCase.query.count()
        return jsonify({'success': True, 'cases': cases, 'total': total}), 200
    except Exception as e:
        logger.exception("Failed to fetch cases")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cases/<int:case_id>/confirm', methods=['POST'])
def confirm_case(case_id):
    """
    Record the actual confirmed fault for a case (mechanic feedback).
    Used to accumulate ground truth data for model retraining.

    Body: {"confirmed_fault": "battery_dead", "notes": "Replaced battery, fixed."}
    """
    try:
        data            = request.get_json(force=True) or {}
        confirmed_fault = data.get('confirmed_fault', '')
        notes           = data.get('notes', '')

        case = DiagnosisCase.query.get(case_id)
        if not case:
            return jsonify({'error': f'Case {case_id} not found.'}), 404

        case.confirmed_fault = confirmed_fault
        case.notes           = notes
        db.session.commit()

        return jsonify({'success': True, 'case': case.to_dict()}), 200
    except Exception as e:
        logger.exception(f"Failed to confirm case {case_id}")
        return jsonify({'error': str(e)}), 500


# ── Model retraining ──────────────────────────────────────────────────────────

@app.route('/api/train', methods=['POST'])
@require_network
def train_model():
    """
    Retrain Bayesian Network parameters from confirmed case history.
    Falls back to synthetic data if no confirmed cases exist.

    Body: {"use_synthetic": true, "n_synthetic": 100}
    """
    try:
        data          = request.get_json(force=True) or {}
        use_synthetic = data.get('use_synthetic', False)
        n_synthetic   = min(int(data.get('n_synthetic', 100)), 500)

        net   = get_network()
        df    = get_cases_as_dataframe() if not use_synthetic else None

        if df is None and not use_synthetic:
            return jsonify({
                'success': False,
                'message': 'No confirmed cases found. Pass "use_synthetic": true to train on synthetic data.',
            }), 400

        if df is None and use_synthetic:
            success = net.train_from_cases(data=None, n_synthetic=n_synthetic)
            source  = f'synthetic ({n_synthetic} cases)'
        else:
            success = net.train_from_cases(data=df)
            source  = f'confirmed cases ({len(df)} records)'

        if success:
            return jsonify({
                'success': True,
                'message': f'Model retrained from {source}.',
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Training failed. Check server logs.',
            }), 500

    except Exception as e:
        logger.exception("Model training failed")
        return jsonify({'error': str(e)}), 500


# ── Symptom description ───────────────────────────────────────────────────────

@app.route('/api/symptom-description/<symptom_id>', methods=['GET'])
@require_network
def symptom_description(symptom_id):
    net   = get_network()
    metas = net.get_symptom_labels()
    if symptom_id not in metas:
        return jsonify({'error': f'Unknown symptom: {symptom_id}', 'valid': list(metas.keys())}), 404
    return jsonify({'symptom': symptom_id, **metas[symptom_id]}), 200


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Smart Car Diagnostic API")
    print("=" * 60)
    print("  GET  /api/health")
    print("  GET  /api/symptoms")
    print("  GET  /api/network")
    print("  POST /api/diagnose")
    print("  POST /api/diagnose/quick")
    print("  POST /api/diagnose/confidence")
    print("  POST /api/next-symptom")
    print("  GET  /api/cases")
    print("  POST /api/cases/<id>/confirm")
    print("  POST /api/train")
    print("=" * 60)
    # Eagerly load network so first request is fast
    get_network()
    app.run(debug=True, host='0.0.0.0', port=5000)
