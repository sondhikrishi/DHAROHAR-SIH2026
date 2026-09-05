"""
=============================================================================
NATIONAL LAND RECORD MODERNIZATION & VALIDATION SYSTEM (DHAROHAR)
Flask REST API Backend (Starter Template)
=============================================================================
Installation:
    pip install -r requirements.txt

Running the Server:
    python app.py
    Server will start at: http://127.0.0.1:5000
=============================================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
# Enable CORS so browser requests from file:// or localhost can communicate seamlessly
CORS(app)

# -----------------------------------------------------------------------------
# 1. IN-MEMORY LAND RECORDS DATABASE
# -----------------------------------------------------------------------------
LAND_RECORDS = [
    {
        "id": "LR-2024-001",
        "khasraNo": "304/1",
        "khatauniNo": "KH-0089",
        "deedNo": "DEED-2021-9981",
        "ownerName": "Ramesh Kumar Sharma",
        "fatherName": "Late Sh. Badri Prasad Sharma",
        "district": "Jaipur Rural",
        "tehsil": "Rampur",
        "village": "Kishorpura",
        "areaAcres": 3.45,
        "soilType": "Agricultural (Irrigated)",
        "marketValue": "₹ 48,50,000",
        "regDate": "2021-03-15",
        "status": "Verified",
        "encumbranceStatus": "Nil / Clear Title",
        "boundaries": "North: Canal Road, South: Plot 303, East: Village Road, West: Plot 304/2"
    },
    {
        "id": "LR-2024-002",
        "khasraNo": "112/A",
        "khatauniNo": "KH-0142",
        "deedNo": "DEED-2022-3118",
        "ownerName": "Sunita Devi",
        "fatherName": "W/o Late Om Prakash Verma",
        "district": "Jaipur Rural",
        "tehsil": "Sadar",
        "village": "Rampur",
        "areaAcres": 2.50,
        "soilType": "Agricultural (Dry Crop)",
        "marketValue": "₹ 35,00,000",
        "regDate": "2019-11-12",
        "status": "Flagged",
        "encumbranceStatus": "Dispute Flagged / Overlapping Claim",
        "boundaries": "North: Panchayat Pathway, South: River Stream, East: Plot 113, West: Plot 111"
    },
    {
        "id": "LR-2024-003",
        "khasraNo": "88/1",
        "khatauniNo": "KH-0056",
        "deedNo": "DEED-2023-4412",
        "ownerName": "Rajesh Dattatray Patil",
        "fatherName": "Sh. Dattatray Patil",
        "district": "Pune North",
        "tehsil": "Haveli",
        "village": "Shivajinagar",
        "areaAcres": 4.10,
        "soilType": "Residential Plot",
        "marketValue": "₹ 1,20,00,000",
        "regDate": "2023-08-20",
        "status": "Flagged",
        "encumbranceStatus": "Excess Area Encroachment Warning",
        "boundaries": "North: State Highway 4, South: Canal, East: Survey 89, West: Survey 87"
    }
]

DUPLICATE_ALERTS = [
    {
        "id": "ALT-2024-01",
        "khasraNo": "112/A",
        "severity": "CRITICAL",
        "title": "High-Risk Overlapping Khasra Claim (Double Registration Fraud)",
        "description": "Two distinct sale deeds have been registered for the exact same parcel within 6 months without succession mutation.",
        "primaryClaimant": {
            "name": "Sunita Devi",
            "deedNo": "DEED-2019-1082",
            "area": "2.50 Acres",
            "date": "12-Nov-2019"
        },
        "conflictingClaimant": {
            "name": "Rajesh Verma",
            "deedNo": "DEED-2024-5501",
            "area": "2.50 Acres",
            "date": "02-Feb-2024"
        },
        "location": "Village Rampur, Tehsil Sadar, Jaipur Rural",
        "status": "Action Required"
    }
]

AUDIT_LOGS = [
    {
        "id": "LOG-901",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "userId": "PAT-402",
        "role": "Patwari",
        "action": "API Server Booted",
        "recordRef": "System Initialized",
        "ipAddress": "127.0.0.1",
        "status": "Success"
    }
]

# -----------------------------------------------------------------------------
# 2. REST API ENDPOINTS
# -----------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def api_root():
    return jsonify({
        "status": "online",
        "service": "DLIMS Land Records REST API",
        "version": "1.0.4",
        "endpoints": [
            "POST /api/login",
            "GET  /api/records",
            "GET  /api/records/<id>",
            "POST /api/records",
            "POST /api/upload-ocr",
            "POST /api/validate/<id>",
            "GET  /api/duplicates",
            "GET  /api/audit-logs",
            "POST /api/audit-logs"
        ]
    })

# 1. User Login / Role Authentication
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "REV-OFFICER-441")
    role = data.get("role", "patwari")

    return jsonify({
        "success": True,
        "token": "bearer-token-dlims-sample-xyz",
        "user": {
            "username": username,
            "role": role,
            "name": "Shri R. K. Sharma" if role == "patwari" else "Authorized Officer"
        }
    })

# 2. Retrieve all land records with optional filtering
@app.route("/api/records", methods=["GET"])
def get_records():
    khasra = request.args.get("khasra", "").lower()
    owner = request.args.get("owner", "").lower()
    district = request.args.get("district", "")

    results = LAND_RECORDS
    if khasra:
        results = [r for r in results if khasra in r["khasraNo"].lower()]
    if owner:
        results = [r for r in results if owner in r["ownerName"].lower()]
    if district and district != "ALL":
        results = [r for r in results if r["district"] == district]

    return jsonify({
        "success": True,
        "count": len(results),
        "records": results
    })

# 3. Retrieve single record by ID
@app.route("/api/records/<record_id>", methods=["GET"])
def get_record_by_id(record_id):
    record = next((r for r in LAND_RECORDS if r["id"] == record_id), None)
    if not record:
        return jsonify({"success": False, "message": "Record not found"}), 404
    return jsonify({"success": True, "record": record})

# 4. Create new land record in registry
@app.route("/api/records", methods=["POST"])
def create_record():
    data = request.get_json()
    if not data or "khasraNo" not in data:
        return jsonify({"success": False, "message": "Missing required field: khasraNo"}), 400

    # Duplicate detection rule check
    is_duplicate = any(r["khasraNo"].lower() == data["khasraNo"].lower() for r in LAND_RECORDS)
    status = "Flagged" if is_duplicate else "Verified"

    new_record = {
        "id": f"LR-2024-{len(LAND_RECORDS) + 1:03d}",
        "khasraNo": data["khasraNo"],
        "khatauniNo": data.get("khatauniNo", "KH-NEW"),
        "deedNo": data.get("deedNo", "DEED-NEW"),
        "ownerName": data.get("ownerName", "Unnamed"),
        "fatherName": data.get("fatherName", "Not specified"),
        "district": data.get("district", "Jaipur Rural"),
        "tehsil": data.get("tehsil", "Sadar"),
        "village": data.get("village", "Rampur"),
        "areaAcres": float(data.get("areaAcres", 1.0)),
        "soilType": data.get("soilType", "Agricultural (Irrigated)"),
        "marketValue": data.get("marketValue", "₹ 25,00,000"),
        "regDate": datetime.now().strftime("%Y-%m-%d"),
        "status": status,
        "encumbranceStatus": "Dispute Flagged" if is_duplicate else "Nil / Verified",
        "boundaries": data.get("boundaries", "Not specified")
    }

    LAND_RECORDS.insert(0, new_record)

    # Append to audit logs
    AUDIT_LOGS.insert(0, {
        "id": f"LOG-{1000 + len(AUDIT_LOGS) + 1}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "userId": "PAT-402",
        "role": "Patwari",
        "action": "New Record Committed",
        "recordRef": f"Khasra #{new_record['khasraNo']}",
        "ipAddress": request.remote_addr or "127.0.0.1",
        "status": "Success"
    })

    return jsonify({
        "success": True,
        "message": "Land record successfully committed.",
        "record": new_record,
        "duplicate_warning": is_duplicate
    }), 201

# 5. Document Upload & OCR Text Extraction
@app.route("/api/upload-ocr", methods=["POST"])
def upload_ocr():
    khasra_target = request.form.get("khasra_no", "304/1")
    doc_type = request.form.get("doc_category", "Registered Sale Deed")

    # In production, you would run:
    # import pytesseract; text = pytesseract.image_to_string(uploaded_file)
    extracted_metadata = {
        "khasraNo": khasra_target,
        "khatauniNo": "KH-0912",
        "deedNo": "DEED-OCR-2024",
        "ownerName": "Shri Mahaveer Prasad Meena",
        "fatherName": "Sh. Ramkaran Meena",
        "areaAcres": 3.80,
        "soilType": "Agricultural (Irrigated)",
        "marketValue": "₹ 45,00,000",
        "village": "Rampur",
        "tehsil": "Sadar",
        "district": "Jaipur Rural",
        "confidence": 98.4,
        "stampDutyVerified": True,
        "boundaries": "North: PWD Road, South: Plot 304, East: Canal, West: Survey 112"
    }

    return jsonify({
        "success": True,
        "docType": doc_type,
        "extracted": extracted_metadata
    })

# 6. Validate Record Integrity
@app.route("/api/validate/<record_id>", methods=["POST"])
def validate_record(record_id):
    rec = next((r for r in LAND_RECORDS if r["id"] == record_id), None)
    if not rec:
        return jsonify({"success": False, "message": "Record not found"}), 404

    # Check for duplicate
    has_duplicates = any(r["khasraNo"].lower() == rec["khasraNo"].lower() and r["id"] != rec["id"] for r in LAND_RECORDS)

    return jsonify({
        "success": True,
        "recordId": record_id,
        "checklist": {
            "aadhaarKyc": "PASSED",
            "stampDutyGRN": "PASSED",
            "nonEncumbrance": "PASSED",
            "cadastralBoundaryOverlap": "FAILED" if has_duplicates else "PASSED",
            "priorMutationClearance": "PASSED"
        },
        "isDuplicate": has_duplicates
    })

# 7. Get Duplicate Flags
@app.route("/api/duplicates", methods=["GET"])
def get_duplicates():
    return jsonify({
        "success": True,
        "count": len(DUPLICATE_ALERTS),
        "duplicates": DUPLICATE_ALERTS
    })

# 8. Audit Logs
@app.route("/api/audit-logs", methods=["GET", "POST"])
def audit_logs():
    if request.method == "POST":
        data = request.get_json() or {}
        new_log = {
            "id": f"LOG-{1000 + len(AUDIT_LOGS) + 1}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "userId": data.get("userId", "API-USER"),
            "role": data.get("role", "Officer"),
            "action": data.get("action", "General Action"),
            "recordRef": data.get("recordRef", "-"),
            "ipAddress": request.remote_addr or "127.0.0.1",
            "status": "Success"
        }
        AUDIT_LOGS.insert(0, new_log)
        return jsonify({"success": True, "log": new_log}), 201

    return jsonify({
        "success": True,
        "count": len(AUDIT_LOGS),
        "logs": AUDIT_LOGS
    })

if __name__ == "__main__":
    print("===============================================================")
    print("  DLIMS Flask REST API Running at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop.")
    print("===============================================================")
    app.run(host="127.0.0.1", port=5000, debug=True)
