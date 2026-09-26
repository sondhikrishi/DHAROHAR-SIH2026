import os
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt
from auth import auth_bp
from models import db, LandRecord, Document, OCRResult
from ocr_service import extract_text
app = Flask(__name__)

# ============================================================
# JWT CONFIGURATION
# ============================================================

app.config["JWT_SECRET_KEY"] = "change-this-to-a-strong-secret-key"

jwt = JWTManager(app)

# Allow frontend to communicate with Flask
CORS(app)

# PostgreSQL connection
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql+psycopg2://postgres:Sondhi%402006@localhost:5432/dharohar"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
db.init_app(app)

# Register authentication routes
app.register_blueprint(auth_bp)

def require_role(*allowed_roles):

    claims = get_jwt()
    user_role = claims.get("role")

    if user_role not in allowed_roles:
        return jsonify({
            "status": "error",
            "message": "You do not have permission to perform this action",
            "required_roles": list(allowed_roles)
        }), 403

    return None

# ============================================================
# CURRENT USER
# ============================================================

@app.route("/api/auth/me", methods=["GET"])
@jwt_required()
def get_current_user():

    claims = get_jwt()

    return jsonify({
        "status": "success",
        "user": {
            "id": claims["sub"],
            "email": claims.get("email"),
            "role": claims.get("role")
        }
    }), 200

# ============================================================
# FILE UPLOAD CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "pdf"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.route("/api/documents/upload", methods=["POST"])
@jwt_required()
def upload_document():

    claims = get_jwt()
    user_id = int(claims["sub"])

    if "file" not in request.files:
        return jsonify({
            "message": "No file provided",
            "status": "error"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "message": "No file selected",
            "status": "error"
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "message": "Unsupported file type",
            "allowed_types": list(ALLOWED_EXTENSIONS),
            "status": "error"
        }), 400

    filename = secure_filename(file.filename)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(file_path)

    file_size = os.path.getsize(file_path)

    document = Document(
        filename=filename,
        file_path=file_path,
        file_type=file.content_type or "unknown",
        file_size=file_size,
        upload_status="Uploaded",
        uploaded_by=user_id
    )

    db.session.add(document)
    db.session.commit()

    return jsonify({
        "message": "Document uploaded successfully",
        "status": "success",
        "document": {
            "id": document.id,
            "filename": document.filename,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "upload_status": document.upload_status
        }
    }), 201


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "DHAROHAR backend is running",
        "status": "success"
    })

# ============================================================
# OCR PROCESSING
# ============================================================

@app.route("/api/documents/<int:document_id>/ocr", methods=["POST"])
@jwt_required()
def process_document_ocr(document_id):

    # --------------------------------------------------------
    # Find document
    # --------------------------------------------------------

    document = db.session.get(Document, document_id)

    if not document:
        return jsonify({
            "message": "Document not found",
            "status": "error"
        }), 404

    # --------------------------------------------------------
    # Check file exists
    # --------------------------------------------------------

    if not os.path.exists(document.file_path):
        return jsonify({
            "message": "Document file not found on server",
            "status": "error"
        }), 404

    try:

        # ----------------------------------------------------
        # Update status
        # ----------------------------------------------------

        document.upload_status = "Processing"
        db.session.commit()

        # ----------------------------------------------------
        # Run OCR
        # ----------------------------------------------------

        extracted_text, confidence = extract_text(
            document.file_path
        )

        # ----------------------------------------------------
        # Check OCR result
        # ----------------------------------------------------

        if not extracted_text.strip():

            document.upload_status = "OCR Failed"
            db.session.commit()

            return jsonify({
                "message": "No text could be extracted",
                "status": "error",
                "document_id": document.id,
                "confidence": 0
            }), 422

        # ----------------------------------------------------
        # Convert confidence to percentage
        # ----------------------------------------------------

        confidence_percentage = round(
            confidence * 100,
            2
        )

        # ----------------------------------------------------
        # Save OCR result
        # ----------------------------------------------------

        ocr_result = OCRResult(
            document_id=document.id,
            extracted_text=extracted_text,
            processing_status="Completed",
            confidence_score=confidence_percentage
        )

        db.session.add(ocr_result)

        # ----------------------------------------------------
        # Update document status
        # ----------------------------------------------------

        document.upload_status = "OCR Completed"

        db.session.commit()

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({
            "message": "OCR completed successfully",
            "status": "success",

            "document": {
                "id": document.id,
                "filename": document.filename,
                "upload_status": document.upload_status
            },

            "ocr": {
                "id": ocr_result.id,
                "confidence": float(
                    ocr_result.confidence_score
                ),
                "extracted_text": ocr_result.extracted_text,
                "processing_status": (
                    ocr_result.processing_status
                )
            }
        }), 200

    except Exception as e:

        db.session.rollback()

        document.upload_status = "OCR Failed"
        db.session.commit()

        return jsonify({
            "message": "OCR processing failed",
            "status": "error",
            "error": str(e)
        }), 500

# ============================================================
# GET OCR RESULT
# ============================================================

@app.route("/api/documents/<int:document_id>/ocr", methods=["GET"])
@jwt_required()
def get_ocr_result(document_id):

    # --------------------------------------------------------
    # Find document
    # --------------------------------------------------------

    document = db.session.get(Document, document_id)

    if not document:
        return jsonify({
            "status": "error",
            "message": "Document not found"
        }), 404

    # --------------------------------------------------------
    # Find latest OCR result
    # --------------------------------------------------------

    ocr_result = (
        OCRResult.query
        .filter_by(document_id=document_id)
        .order_by(OCRResult.id.desc())
        .first()
    )

    if not ocr_result:
        return jsonify({
            "status": "error",
            "message": "OCR result not found"
        }), 404

    # --------------------------------------------------------
    # Return OCR result
    # --------------------------------------------------------

    return jsonify({
        "status": "success",

        "document": {
            "id": document.id,
            "filename": document.filename,
            "upload_status": document.upload_status
        },

        "ocr": {
            "id": ocr_result.id,
            "extracted_text": ocr_result.extracted_text,
            "confidence": float(
                ocr_result.confidence_score
            ),
            "processing_status": ocr_result.processing_status,
            "processed_at": (
                ocr_result.processed_at.isoformat()
                if ocr_result.processed_at
                else None
            )
        }
    }), 200
# ============================================================
# GET ALL LAND RECORDS
# ============================================================

@app.route("/api/land-records", methods=["GET"])
@jwt_required()
def get_land_records():

    records = LandRecord.query.all()

    return jsonify([
        {
            "id": record.id,
            "khasra_no": record.khasra_no,
            "khatauni_no": record.khatauni_no,
            "deed_no": record.deed_no,
            "owner_name": record.owner_name,
            "father_name": record.father_name,
            "village": record.village,
            "tehsil": record.tehsil,
            "district": record.district,
            "area_acres": (
                float(record.area_acres)
                if record.area_acres is not None
                else None
            ),
            "soil_type": record.soil_type,
            "market_value": record.market_value,
            "registration_date": (
                record.registration_date.isoformat()
                if record.registration_date
                else None
            ),
            "status": record.status,
            "encumbrance_status": record.encumbrance_status,
            "boundaries": record.boundaries,
            "created_at": (
                record.created_at.isoformat()
                if record.created_at
                else None
            ),
            "updated_at": (
                record.updated_at.isoformat()
                if record.updated_at
                else None
            )
        }
        for record in records
    ])


# ============================================================
# GET SINGLE LAND RECORD
# ============================================================

@app.route("/api/land-records/<int:record_id>", methods=["GET"])
@jwt_required()
def get_land_record(record_id):

    record = db.session.get(LandRecord, record_id)

    if not record:
        return jsonify({
            "message": "Land record not found",
            "status": "error"
        }), 404

    return jsonify({
        "id": record.id,
        "khasra_no": record.khasra_no,
        "khatauni_no": record.khatauni_no,
        "deed_no": record.deed_no,
        "owner_name": record.owner_name,
        "father_name": record.father_name,
        "village": record.village,
        "tehsil": record.tehsil,
        "district": record.district,
        "area_acres": (
            float(record.area_acres)
            if record.area_acres is not None
            else None
        ),
        "soil_type": record.soil_type,
        "market_value": record.market_value,
        "registration_date": (
            record.registration_date.isoformat()
            if record.registration_date
            else None
        ),
        "status": record.status,
        "encumbrance_status": record.encumbrance_status,
        "boundaries": record.boundaries
    })


# ============================================================
# CREATE LAND RECORD
# ============================================================

@app.route("/api/land-records", methods=["POST"])
@jwt_required()
def create_land_record():

    permission_error = require_role("patwari", "admin")

    if permission_error:
        return permission_error

    data = request.get_json()

    if not data:
        return jsonify({
            "message": "Request body is required",
            "status": "error"
        }), 400

    required_fields = [
        "khasra_no",
        "owner_name",
        "village",
        "tehsil",
        "district"
    ]

    missing_fields = [
        field for field in required_fields
        if not data.get(field)
    ]

    if missing_fields:
        return jsonify({
            "message": "Missing required fields",
            "missing_fields": missing_fields,
            "status": "error"
        }), 400

    record = LandRecord(
        khasra_no=data["khasra_no"],
        khatauni_no=data.get("khatauni_no"),
        deed_no=data.get("deed_no"),
        owner_name=data["owner_name"],
        father_name=data.get("father_name"),
        village=data["village"],
        tehsil=data["tehsil"],
        district=data["district"],
        area_acres=data.get("area_acres"),
        soil_type=data.get("soil_type"),
        market_value=data.get("market_value"),
        registration_date=data.get("registration_date"),
        status=data.get("status", "Pending"),
        encumbrance_status=data.get("encumbrance_status"),
        boundaries=data.get("boundaries")
    )

    db.session.add(record)
    db.session.commit()

    return jsonify({
        "message": "Land record created successfully",
        "status": "success",
        "record_id": record.id
    }), 201


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )