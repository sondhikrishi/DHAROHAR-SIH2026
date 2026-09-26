from flask import Flask, jsonify
from flask_cors import CORS
from models import db

app = Flask(__name__)
CORS(app)

# PostgreSQL connection
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql+psycopg2://postgres:Sondhi%402006@localhost:5432/dharohar"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
db.init_app(app)


@app.route("/")
def home():
    return jsonify({
        "message": "DHAROHAR backend is running",
        "status": "success"
    })


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)