from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    role = db.Column(
        db.String(20),
        nullable=False,
        default="citizen"
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class LandRecord(db.Model):
    __tablename__ = "land_records"

    id = db.Column(db.Integer, primary_key=True)

    khasra_no = db.Column(db.String(100), nullable=False, index=True)
    khatauni_no = db.Column(db.String(100))
    deed_no = db.Column(db.String(100))

    owner_name = db.Column(db.String(150), nullable=False)
    father_name = db.Column(db.String(150))

    village = db.Column(db.String(100), nullable=False)
    tehsil = db.Column(db.String(100), nullable=False)
    district = db.Column(db.String(100), nullable=False)

    area_acres = db.Column(db.Numeric(10, 2))
    soil_type = db.Column(db.String(100))
    market_value = db.Column(db.String(100))

    registration_date = db.Column(db.Date)

    status = db.Column(
        db.String(30),
        nullable=False,
        default="Pending"
    )

    encumbrance_status = db.Column(db.String(100))
    boundaries = db.Column(db.Text)

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)

    filename = db.Column(
        db.String(255),
        nullable=False
    )

    file_path = db.Column(
        db.String(500),
        nullable=False
    )

    file_type = db.Column(
        db.String(50),
        nullable=False
    )

    file_size = db.Column(db.Integer)

    upload_status = db.Column(
        db.String(30),
        nullable=False,
        default="Uploaded"
    )

    uploaded_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    uploaded_by = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=True
    )

    land_record_id = db.Column(
        db.Integer,
        db.ForeignKey("land_records.id"),
        nullable=True
    )


class OCRResult(db.Model):
    __tablename__ = "ocr_results"

    id = db.Column(db.Integer, primary_key=True)

    document_id = db.Column(
        db.Integer,
        db.ForeignKey("documents.id"),
        nullable=False
    )

    extracted_text = db.Column(
        db.Text,
        nullable=False
    )

    processing_status = db.Column(
        db.String(30),
        nullable=False,
        default="Completed"
    )

    confidence_score = db.Column(
        db.Numeric(5, 2)
    )

    processed_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )