from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure the SQLAlchemy part of the app
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/Face'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create the SQLAlchemy object
db = SQLAlchemy(app)

# Define your models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), index=True)
    email = db.Column(db.String(100), unique=True, index=True)
    face_embedding = db.Column(db.LargeBinary)

# Create the database tables
with app.app_context():
    db.create_all()
