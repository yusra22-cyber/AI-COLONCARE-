import os
import traceback
import logging
import uuid
from functools import wraps
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory, redirect, session, flash
from flask import abort
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect, generate_csrf
import torch
from PIL import Image
from torchvision import transforms
import timm
import re
from flask_limiter import Limiter
from flask_mail import Mail, Message
from flask_limiter.util import get_remote_address
from bleach import clean
from dotenv import load_dotenv
from flask_migrate import Migrate
from skimage import exposure
import numpy as np
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import ImageEnhance

load_dotenv() 
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid")

# --- Flask App Configuration ---
app = Flask( __name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = '5ce918798d862603911547c7da22803b0d5ac14a26478ef3b7421ed011cdcbb9'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024  # 24MB limit
app.config['ALLOWED_EXTENSIONS'] = { 'png', 'jpg', 'jpeg'}
app.config['SECURITY_PASSWORD_HASH'] = 'pbkdf2_sha512'
app.config['SECURITY_PASSWORD_SALT'] = os.environ.get('SECURITY_PASSWORD_SALT')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['WTF_CSRF_ENABLED'] = True  
   
  # Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

mail = Mail(app)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"  # More consistent rate limiting
    
)
# Later when you create the app:
limiter.init_app(app)
# Security configurations
app.config.update({
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///users.db',
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'UPLOAD_FOLDER': 'static/uploads',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg'},
    'WTF_CSRF_ENABLED': True,
    'SESSION_COOKIE_SECURE': True,
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Lax',
    'SESSION_REFRESH_EACH_REQUEST': True,
    'PERMANENT_SESSION_LIFETIME': timedelta(minutes=30),
    'WTF_CSRF_TIME_LIMIT': 3600

})

# Initialize extensions
db = SQLAlchemy(app)
CORS(app, supports_credentials=True)
csrf = CSRFProtect(app)
migrate = Migrate(app, db)



# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    failed_login_attempts = db.Column(db.Integer, default=0)  
    account_locked = db.Column(db.Boolean, default=False) 
    last_login_attempt = db.Column(db.DateTime) 
    is_admin = db.Column(db.Boolean, default=False) 
    last_password_change = db.Column(db.DateTime)
    security_notifications = db.relationship('SecurityNotification', backref='user', lazy=True)

class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    activity_type = db.Column(db.String(50))  # 'login_success', 'login_failure', 'logout'
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=db.func.now())
    details = db.Column(db.Text)  # Additional context if needed
class PredictionResult(db.Model):
    __tablename__ = 'prediction_results'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    image_url = db.Column(db.String(500))
    image_name = db.Column(db.String(200))
    prediction = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    all_predictions = db.Column(db.JSON)  # Store all class probabilities
    is_colonoscopy = db.Column(db.Boolean)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='predictions')
# --- Model Loading ---
def load_polyp_model():
    try:
        MODEL_DIR = 'model'
        MODEL_FILE = 'best_model-classification.pth'
        MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, MODEL_FILE))
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        # Initialize with timm 0.9.2 compatible settings
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=5,
           
        )
        
        # Load checkpoint with error tolerance
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model_state_dict = model.state_dict()
        matched_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
        
        model.load_state_dict(matched_state_dict, strict=False)
       
        
        # Verify model
        model.eval()
        
        return model

    except Exception as e:
        logger.error(f"Polyp Model loading failed: {str(e)}")
        traceback.print_exc()
        raise


def load_binary_model():
    try:
        BINARY_MODEL_DIR = 'model'
        BINARY_MODEL_FILE = 'binary_mobilenetv3_best.pth'
        BINARY_MODEL_PATH = os.path.abspath(os.path.join(BINARY_MODEL_DIR, BINARY_MODEL_FILE))
        
        if not os.path.exists(BINARY_MODEL_PATH):
            raise FileNotFoundError(f"Binary model file not found at: {BINARY_MODEL_PATH}")

        # Initialize model with updated syntax
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', weights=None)
        
        # Modify classifier for binary classification
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
        
        # Load checkpoint
        checkpoint = torch.load(BINARY_MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        
        return model

    except Exception as e:
        logger.error(f"Binary Model loading failed: {str(e)}")
        traceback.print_exc()
        raise
   
try:
    POLYP_CLASS_NAMES = ['Hyperplastic', 'Serrated', 'Tubular', 'Tubulovillous', 'Villous']
    BINARY_CLASS_NAMES = ['Colonoscopy', 'Non-Colonoscopy']
    
    polyp_model = load_polyp_model()
    binary_model = load_binary_model()
    
except Exception as e:
    logger.error(f"Critical initialization error: {str(e)}")
    raise RuntimeError("Failed to initialize models") from e
# --- Image Preprocessing ---
polyp_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

binary_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404
     

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# --- Routes ---
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if request.method == 'POST':
        # Check if request wants JSON response
        wants_json = request.accept_mimetypes['application/json']
        
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        # Record login attempt
        if user:
            record_login_activity(user.id, 'login_attempt', request)
        else:
            activity = UserActivity(
                activity_type='login_attempt',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string,
                details=f"Attempt with non-existent email: {email}"
            )
            db.session.add(activity)
            db.session.commit()

        if user:
            if user.account_locked:
                remaining_time = (user.last_login_attempt + timedelta(minutes=30) - datetime.now()).total_seconds() / 60
                if remaining_time > 0:
                    if wants_json:
                        return jsonify({
                            'success': False,
                            'error': f'Account locked. Try again in {int(remaining_time)} minutes'
                        }), 403
                    flash(f'Account locked. Try again in {int(remaining_time)} minutes', 'danger')
                    return redirect(url_for('login'))

            if check_password_hash(user.password, password):
                # Successful login
                user.failed_login_attempts = 0
                db.session.commit()
                
                session['user_id'] = user.id
                session['user_email'] = user.email
                session['user_name'] = f"{user.first_name} {user.last_name}"
                
                record_login_activity(user.id, 'login_success', request)
                
                if wants_json:
                    return jsonify({
                        'success': True,
                        'redirect': url_for('website')
                    })
                flash('Login successful!', 'success')
                return redirect(url_for('website'))
            else:
                # Failed login
                user.failed_login_attempts += 1
                user.last_login_attempt = db.func.now()
                db.session.commit()
                record_login_activity(user.id, 'login_failure', request)
                
                error_msg = 'Account locked due to too many failed attempts' if check_login_security(user) else 'Invalid email or password'
                
                if wants_json:
                    return jsonify({
                        'success': False,
                        'error': error_msg
                    }), 401
                flash(error_msg, 'danger')
        else:
            error_msg = 'Invalid email or password'
            if wants_json:
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 401
            flash(error_msg, 'danger')

        return redirect(url_for('login'))

    return render_template('login.html', register=False)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('firstName')
        last_name = request.form.get('lastName')
        email = request.form.get('emailCreate')
        password = request.form.get('passwordCreate')

        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=hashed_password
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration error: {str(e)}")
            flash('Registration failed. Please try again.', 'danger')
            return redirect(url_for('register'))

    return render_template('login.html', register=True)

@app.route('/logout', methods=['POST'])
def logout():
    if 'user_id' in session:
        record_login_activity(session['user_id'], 'logout', request)
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))
@app.route('/login-history')
@login_required
def login_history():
    activities = UserActivity.query.filter_by(user_id=session['user_id'])\
        .order_by(UserActivity.timestamp.desc())\
        .limit(50)\
        .all()
    return render_template('login_history.html', activities=activities)    

@app.route('/validate-session')
def validate_session():
    if 'user_id' in session:
        return jsonify({
            'valid': True,
            'user': session.get('user_name')
        })
    return jsonify({'valid': False}), 401

@app.route('/website')
@login_required
def website():
    return render_template('website.html', username=session.get('user_name'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('file.html')

@app.route('/profile')
@login_required
def profile():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    
    # Get user's prediction history with pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    predictions = PredictionResult.query.filter_by(user_id=user_id)\
        .order_by(PredictionResult.timestamp.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('updated_abbb.html', predictions=predictions)

@app.route('/settings')
@login_required
def settings():
    return render_template('setting.html')

@app.route('/indexx')
def indexx():
    return render_template('index.html')
def record_login_activity(user_id, activity_type, request):
    """Record login activity in database"""
    try:
        activity = UserActivity(
            user_id=user_id,
            activity_type=activity_type,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string,
            details=f"{activity_type} attempt"
        )
        db.session.add(activity)
        db.session.commit()
    except Exception as e:
        logger.error(f"Failed to record login activity: {str(e)}")
        db.session.rollback()
class SecurityNotification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    notification_type = db.Column(db.String(50))
    message = db.Column(db.Text)
    read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=db.func.now())

def check_login_security(user):
    """Check if account should be locked due to failed attempts"""
    if user.failed_login_attempts >= 5:
        user.account_locked = True
        user.last_login_attempt = db.func.now()
        db.session.commit()
        record_login_activity(user.id, 'account_locked', request)
        return True
    return False

def check_login_security(user, request):  # Add request parameter
    if user.failed_login_attempts >= 5:
        user.account_locked = True
        user.last_login_attempt = db.func.now()
        db.session.commit()
        record_login_activity(user.id, 'account_locked', request)
        return True
    return False

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
@login_required
@limiter.limit("100/day")
def predict():
    logger.info("--- /predict endpoint called ---")
    
    if 'file' not in request.files:
        logger.error("No 'file' key in request.files")
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        logger.error(f"Invalid file: {file.filename}")
        return jsonify({'success': False, 'error': 'Invalid file'}), 400

    try:
        # Save file
        upload_dir = os.path.join(app.static_folder, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # Verify image
        img = Image.open(filepath).convert("RGB")
        img.verify()
        img = Image.open(filepath).convert("RGB")

        # Binary classification - first check if it's a colonoscopy image
        binary_input = binary_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            binary_output = binary_model(binary_input)
            binary_probs = torch.nn.functional.softmax(binary_output, dim=1)
            binary_conf, binary_pred = torch.max(binary_probs, 1)
            binary_class = BINARY_CLASS_NAMES[binary_pred.item()]
            binary_confidence = float(binary_conf.item())

        # If NOT a colonoscopy image, return immediately with rejection
        if binary_class != 'Colonoscopy':
            # Clean up the uploaded non-colonoscopy image
            os.remove(filepath)
            logger.info("Non-colonoscopy image removed")
            
            return jsonify({
                'success': False,
                'error': 'Invalid image type',
                'message': 'The uploaded image is not a colonoscopy image. Please upload a valid colonoscopy image.',
                'binary_prediction': binary_class,
                'binary_confidence': binary_confidence
            }), 400

        # Only proceed with polyp classification if it's a colonoscopy image
        polyp_input = polyp_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            polyp_output = polyp_model(polyp_input)
            polyp_probs = torch.nn.functional.softmax(polyp_output, dim=1)
            polyp_conf, polyp_pred = torch.max(polyp_probs, 1)
            polyp_class = POLYP_CLASS_NAMES[polyp_pred.item()]
            polyp_confidence = float(polyp_conf.item())
        
        result_data = {
            'success': True,
            'is_colonoscopy': True,
            'binary_prediction': binary_class,
            'binary_confidence': binary_confidence,
            'polyp_prediction': polyp_class,
            'polyp_confidence': polyp_confidence,
            'image_url': f"/static/uploads/{filename}",
            'image_name': filename,
            'timestamp': datetime.now().isoformat(),
            
        }

        store_result(result_data)
        
        # Add CORS headers to response
        response = jsonify(result_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        
        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        # Clean up if file was created
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({
            'success': False,
            'error': 'Image processing failed',
            'details': str(e) if app.debug else None
        }), 500
def store_result(result_data):
    """Store prediction results in database with proper image handling"""
    try:
        if 'user_id' in session:
            new_result = PredictionResult(
                user_id=session['user_id'],
                image_url=result_data['image_url'],
                image_name=os.path.basename(result_data['image_url']),
                prediction=result_data.get('polyp_prediction', result_data['binary_prediction']),
                confidence=result_data.get('polyp_confidence', result_data['binary_confidence']),
                all_predictions=result_data.get('all_predictions', {}),
                is_colonoscopy=result_data['is_colonoscopy'],
                timestamp=datetime.now()
            )
            db.session.add(new_result)
            db.session.commit()
            return True
        return False
    except Exception as e:
        logger.error(f"Result storage failed: {str(e)}")
        db.session.rollback()
        return False


@app.route('/test-email')
def test_email():
    try:
        msg = Message('Test Email', 
                     recipients=['recipient@example.com'])
        msg.body = "This is a test email from Flask"
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {str(e)}"
@app.after_request
def set_csrf_cookie(response):
    response.set_cookie('csrf_token', generate_csrf())
    return response

def add_security_headers(response):
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# Settings Routes
@app.route('/update_settings', methods=['POST'])
@login_required
def update_settings():
    try:
        user_id = session.get('user_id')
        logger.info(f"User {user_id} attempting settings update")
        if not user_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401

        data = request.get_json()
        
        # Validate email format
        if 'email' in data:
            if not re.match(r"[^@]+@[^@]+\.[^@]+", data['email']):
                return jsonify({'success': False, 'error': 'Invalid email format'}), 400
        
        # Update user settings in database
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
            
        if 'email' in data:
            user.email = data['email']
        if 'notification_preference' in data:
            user.notification_preference = data['notification_preference']
        if 'password' in data and data['password']:
            user.password = generate_password_hash(data['password'])
            
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Settings update error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/admin')
@login_required
def admin_dashboard():
    # Check if user is logged in
    if 'user_id' not in session:
        abort(401)  # Unauthorized
    
    # Get current user
    user = User.query.get(session['user_id'])
    if not user:
        abort(401)  # Unauthorized
    
    # Check admin status
    if not user.is_admin:
        abort(403)  # Forbidden
    
    # Create admin user if none exists (only if current user is admin)
    if not User.query.filter_by(is_admin=True).first():
        admin_email = os.environ.get('ADMIN_EMAIL')
        admin_password = os.environ.get('ADMIN_PASSWORD')
        
        if not admin_email or not admin_password:
            flash('Admin credentials not configured', 'error')
            abort(500)
        
        admin = User(
            first_name='Admin',
            last_name='User',
            email=admin_email,
            password=generate_password_hash(admin_password),
            is_admin=True
        )
        db.session.add(admin)
        try:
            db.session.commit()
            flash('Default admin user created', 'success')
        except Exception as e:
            db.session.rollback()
            flash('Failed to create admin user', 'error')
            abort(500)
    
    return render_template('admin_dashboard.html')
@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    try:
        user_id = session['user_id']
        user = User.query.get(user_id)
        
        if not user:
            return jsonify(success=False, error="User not found"), 404
        
        data = request.get_json()
        if not data or 'password' not in data:
            return jsonify(success=False, error="Password required"), 400
        
        if not check_password_hash(user.password, data['password']):
            return jsonify(success=False, error="Invalid password"), 401
        
        db.session.delete(user)
        db.session.commit()
        session.clear()  # Critical: Logs user out
        
        return jsonify(success=True)
    
    except Exception as e:
        db.session.rollback()
        logger.error(f"Delete error: {e}")
        return jsonify(success=False, error="Server error"), 500
    
def is_email_configured():
    """Check if email settings are properly configured"""
    required = ['MAIL_SERVER', 'MAIL_USERNAME', 'MAIL_PASSWORD']
    return all(app.config.get(key) for key in required)


@app.route('/submit_contact', methods=['POST'])
@limiter.limit("5 per hour")
def submit_contact():
    try:
        data = request.get_json()
        required_fields = ['name', 'email', 'subject', 'message']
        
        if not all(field in data for field in required_fields):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        # Sanitize inputs
        clean_name = clean(data['name'])
        clean_email = clean(data['email'])
        clean_subject = clean(data['subject'])
        clean_message = clean(data['message'])

        msg = Message(
            subject=f"New Contact: {clean_subject}",
            recipients=[os.getenv('CONTACT_FORM_RECIPIENT')],
            body=f"""
            Name: {clean_name}
            Email: {clean_email}
            Message: {clean_message}
            """
        )
        mail.send(msg)

        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Contact form error: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to send message'}), 500
    

if __name__ == '__main__':
    # Validate email configuration
    if not is_email_configured():
        logger.warning("Email configuration is incomplete - contact form will not work")
    
    with app.app_context():
        db.create_all()
        print("Database tables created!")
    
    
    # Model verification
    try:
        print("Verifying models...")
        test_input = torch.rand(1, 3, 224, 224)
        
        # Verify binary model
        binary_output = binary_model(test_input)
        assert binary_output.shape == (1, 2), f"Binary model output shape mismatch: {binary_output.shape}"
        
        # Verify polyp model
        polyp_output = polyp_model(test_input)
        assert polyp_output.shape == (1, 5), f"Polyp model output shape mismatch: {polyp_output.shape}"
        
        print("Model verification successful!")
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        raise
    
    print(User.__table__.columns.keys())
    app.run(host='0.0.0.0', port=5000, debug=True)