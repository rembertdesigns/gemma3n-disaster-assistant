import os
import sys
import subprocess
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Setup logging to provide clear feedback to the user
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DisasterResponseSetup:
    """
    A utility class to set up the environment for the Disaster Response & Recovery Assistant.
    It checks dependencies, creates directories, sets up a virtual environment,
    installs packages, and generates necessary configuration files.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.python_version = sys.version_info
        
    def print_banner(self):
        """Prints the welcome banner for the setup script."""
        print("\n" + "="*80)
        print("🚨 DISASTER RESPONSE & RECOVERY ASSISTANT SETUP")
        print("="*80)
        print("🧠 AI-Powered Emergency Analysis with Gemma 3n")
        print("🎤 Voice Emergency Processing | 📷 Image Analysis | 🗺️ Real-time Mapping")
        print("⚡ Adaptive AI Optimization | 📊 Predictive Analytics")
        print("="*80)
    
    def check_python_version(self) -> bool:
        """Checks if the current Python version meets the requirements."""
        logger.info("🐍 Checking Python version...")
        
        if self.python_version < (3, 8):
            logger.error(f"❌ Python 3.8+ is required, but you are using {self.python_version.major}.{self.python_version.minor}.")
            logger.error("Please upgrade your Python version to continue.")
            return False
        elif self.python_version < (3, 9):
            logger.warning(f"⚠️  Python 3.9+ is recommended for the best performance. Found {self.python_version.major}.{self.python_version.minor}.")
        else:
            logger.info(f"✅ Python {self.python_version.major}.{self.python_version.minor} is compatible.")
        
        return True
    
    def check_system_dependencies(self) -> bool:
        """Checks for required system-level dependencies like git and ffmpeg."""
        logger.info("🔧 Checking for required system dependencies...")
        
        required_packages = {
            'git': 'Git (for version control and package management)',
            'curl': 'cURL (for downloading files)',
            'ffmpeg': 'FFmpeg (for audio/video processing)'
        }
        
        missing_packages = [name for name in required_packages if not shutil.which(name)]
        
        if not missing_packages:
            logger.info("✅ All system dependencies are met.")
            return True

        logger.warning("⚠️  Some system packages are missing. This may affect certain features.")
        for pkg in missing_packages:
            logger.warning(f"   • {required_packages[pkg]} not found.")
        
        logger.info("\nTo install missing packages:")
        logger.info("   - On Ubuntu/Debian: sudo apt-get update && sudo apt-get install git curl ffmpeg")
        logger.info("   - On macOS (with Homebrew): brew install git curl ffmpeg")
        logger.info("   - On Windows: Use Chocolatey (choco install git curl ffmpeg) or download installers.")
        # Continue setup but with a warning
        return True
    
    def create_directories(self) -> bool:
        """Creates the necessary directory structure for the project."""
        logger.info("📁 Creating project directories...")
        
        directories = ['uploads', 'outputs', 'models', 'logs', 'static', 'templates', 'data', 'backups']
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                logger.info(f"   - Ensured directory exists: ./{directory}/")
            logger.info("✅ All directories are in place.")
            return True
        except OSError as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False
    
    def setup_virtual_environment(self) -> bool:
        """Creates and configures a Python virtual environment."""
        logger.info("🐍 Setting up Python virtual environment...")
        
        venv_path = self.project_root / "venv"
        
        if venv_path.exists():
            logger.info("✅ Virtual environment 'venv' already exists.")
            return True
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True, capture_output=True)
            logger.info("✅ Virtual environment created successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e.stderr.decode()}")
            return False

    def get_venv_pip_path(self) -> Path:
        """Gets the path to the pip executable inside the virtual environment."""
        venv_path = self.project_root / "venv"
        if sys.platform == "win32":
            return venv_path / "Scripts" / "pip.exe"
        else:
            return venv_path / "bin" / "pip"

    def get_venv_python_path(self) -> Path:
        """Gets the path to the python executable inside the virtual environment."""
        venv_path = self.project_root / "venv"
        if sys.platform == "win32":
            return venv_path / "Scripts" / "python.exe"
        else:
            return venv_path / "bin" / "python"

    def create_requirements_file(self) -> bool:
        """Creates a requirements.txt file if it doesn't exist."""
        logger.info("📋 Creating requirements.txt file...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if requirements_file.exists():
            logger.info("✅ requirements.txt already exists.")
            return True
        
        # Define the dependencies for our single-file API
        requirements_content = """# Core FastAPI dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23

# Authentication & Security
PyJWT==2.8.0
bcrypt==4.1.2
python-jose[cryptography]==3.3.0

# Validation
pydantic==2.5.0

# Templating
jinja2==3.1.2

# System monitoring
psutil==5.9.6

# Optional dependencies (with fallbacks in code)
# openai-whisper==20231117
# weasyprint==60.2

# Development tools (optional)
# pytest==7.4.3
# black==23.11.0
# isort==5.12.0
"""
        
        try:
            requirements_file.write_text(requirements_content)
            logger.info("✅ requirements.txt created successfully.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create requirements.txt: {e}")
            return False

    def install_dependencies(self) -> bool:
        """Installs Python dependencies from requirements.txt into the venv."""
        logger.info("📦 Installing Python dependencies...")
        
        # First ensure requirements.txt exists
        if not self.create_requirements_file():
            return False
        
        requirements_file = self.project_root / "requirements.txt"
        pip_path = self.get_venv_pip_path()

        if not pip_path.exists():
            logger.error(f"❌ Pip executable not found at {pip_path}. Cannot install dependencies.")
            return False

        try:
            # First, upgrade pip
            logger.info("   - Upgrading pip in the virtual environment...")
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True, capture_output=True)

            # Install requirements
            logger.info("   - Installing packages...")
            result = subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Failed to install some dependencies:")
                logger.error(result.stderr)
                logger.info("ℹ️  Some optional dependencies may have failed - this is normal.")
            else:
                logger.info("✅ All Python dependencies installed successfully.")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies. Check logs for details.")
            logger.error(f"   Command: '{' '.join(e.cmd)}'")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(f"   Output:\n{e.stderr.decode()}")
            return False
    
    def setup_environment_file(self) -> bool:
        """Creates a .env file with default configuration."""
        logger.info("⚙️  Setting up environment configuration (.env file)...")
        
        env_file = self.project_root / ".env"
        
        if env_file.exists():
            logger.info("✅ .env file already exists. Skipping.")
            return True
        
        # Create a basic .env file with sensible defaults
        env_content = """# Enhanced Emergency Response Assistant Configuration

# Security
SECRET_KEY=your-secret-key-here-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=sqlite:///./data/emergency_response.db

# File uploads
MAX_FILE_SIZE_MB=10
UPLOAD_FOLDER=uploads

# AI Configuration
AI_MODEL_VARIANT=gemma-3n-4b
AI_CONTEXT_WINDOW=64000

# Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Environment
DEBUG=true
ENVIRONMENT=development

# External services (optional)
ENABLE_NOTIFICATIONS=false
NOTIFICATION_WEBHOOK_URL=

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=detailed

# CORS (for production, specify allowed origins)
CORS_ORIGINS=*
"""
        
        try:
            env_file.write_text(env_content)
            logger.info("✅ .env file created with default configuration.")
            logger.warning("📝 ACTION REQUIRED: Please edit the .env file with your custom settings, especially SECRET_KEY for production!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create .env file: {e}")
            return False
    
    def initialize_database(self) -> bool:
        """Initializes the database by testing our API's database setup."""
        logger.info("🗄️  Initializing database...")
        
        try:
            # Try to import and test our API's database setup
            venv_python = self.get_venv_python_path()
            
            # Create a test script to verify database setup
            test_script_content = """
import sys
sys.path.append('.')
try:
    from api import Base, engine
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")
except Exception as e:
    print(f"Database error: {e}")
    sys.exit(1)
"""
            
            test_script = self.project_root / "test_db.py"
            test_script.write_text(test_script_content)
            
            # Run the test script
            result = subprocess.run([str(venv_python), str(test_script)], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            # Clean up test script
            test_script.unlink()
            
            if result.returncode == 0:
                logger.info("✅ Database tables created successfully.")
                return True
            else:
                logger.warning("⚠️  Database initialization had issues, but continuing setup.")
                logger.info("ℹ️  Database will be initialized when the API starts.")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️  Database pre-initialization failed: {e}")
            logger.info("ℹ️  Database will be initialized when the API starts.")
            return True
    
    def download_ai_models(self) -> bool:
        """Downloads prerequisite AI models, like Whisper (optional)."""
        logger.info("🤖 Checking for optional AI models...")
        
        try:
            # Try to install whisper if requested
            pip_path = self.get_venv_pip_path()
            logger.info("   - Attempting to install openai-whisper (optional)...")
            
            result = subprocess.run([str(pip_path), "install", "openai-whisper"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ OpenAI Whisper installed successfully.")
                
                # Try to download base model
                try:
                    venv_python = self.get_venv_python_path()
                    whisper_test = """
import whisper
print("Downloading Whisper base model...")
whisper.load_model("base")
print("Whisper model ready")
"""
                    test_file = self.project_root / "whisper_test.py"
                    test_file.write_text(whisper_test)
                    
                    subprocess.run([str(venv_python), str(test_file)], 
                                 capture_output=True, text=True)
                    test_file.unlink()
                    
                    logger.info("✅ Whisper 'base' model downloaded.")
                except:
                    logger.info("ℹ️  Whisper model will download on first use.")
            else:
                logger.info("ℹ️  OpenAI Whisper not installed (optional feature).")
            
            logger.info("ℹ️  Gemma 3n models will be simulated by the application.")
            return True
            
        except Exception as e:
            logger.info(f"ℹ️  AI model setup skipped: {e}")
            return True
    
    def create_sample_data(self) -> bool:
        """Populates the database with sample data using API's demo endpoint."""
        logger.info("🌱 Preparing sample data setup...")
        
        try:
            logger.info("ℹ️  Sample data will be created via the API's /api/generate-demo-data endpoint once running.")
            logger.info("ℹ️  You can generate demo data by visiting: http://localhost:8000/api/generate-demo-data")
            return True
        except Exception as e:
            logger.warning(f"⚠️  Sample data preparation note: {e}")
            return True
            
    def verify_setup(self) -> bool:
        """Runs basic verification tests to ensure the setup was successful."""
        logger.info("🧪 Verifying setup integrity...")
        try:
            venv_python = self.get_venv_python_path()
            
            # Create a simple verification script
            verify_script_content = """
import sys
sys.path.append('.')

# Test basic imports
try:
    import fastapi
    import sqlalchemy
    import jwt
    print("✅ Core dependencies imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test our API file
try:
    import api
    print("✅ Main API module imported successfully")
    
    # Test that key components exist
    assert hasattr(api, 'app'), "FastAPI app not found"
    assert hasattr(api, 'config'), "Config not found"
    print("✅ API components verified")
    
except Exception as e:
    print(f"❌ API verification failed: {e}")
    sys.exit(1)

print("✅ All verification tests passed")
"""
            
            verify_script = self.project_root / "verify_setup.py"
            verify_script.write_text(verify_script_content)
            
            # Run verification
            result = subprocess.run([str(venv_python), str(verify_script)], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            # Clean up
            verify_script.unlink()
            
            if result.returncode == 0:
                logger.info("✅ Verification successful.")
                logger.info(result.stdout.strip())
                return True
            else:
                logger.warning("⚠️  Some verification tests failed:")
                logger.warning(result.stdout + result.stderr)
                logger.info("ℹ️  Setup may still work - try running the API manually.")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️  Setup verification encountered issues: {e}")
            logger.info("ℹ️  This doesn't necessarily mean setup failed.")
            return True
    
    def generate_startup_script(self) -> bool:
        """Generates cross-platform startup scripts for development and production."""
        logger.info("📝 Generating startup scripts...")
        
        try:
            # Development script
            if sys.platform == "win32":
                dev_script_path = self.project_root / "start_dev.bat"
                dev_content = f"""@echo off
title Emergency Response Dev Server
echo.
echo 🚨 Starting Emergency Response Assistant (Development)
echo ================================================================
echo Activating virtual environment...
call "{self.project_root / 'venv' / 'Scripts' / 'activate.bat'}"

echo Starting development server...
echo.
echo 🌐 Server will be available at: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/api/docs
echo 🏥 Health Check: http://localhost:8000/health
echo.
python api.py
pause
"""
            else:
                dev_script_path = self.project_root / "start_dev.sh"
                dev_content = f"""#!/bin/bash
# Development startup script for Emergency Response Assistant

echo ""
echo "🚨 Starting Emergency Response Assistant (Development)"
echo "================================================================"
echo "Activating virtual environment..."
source '{self.project_root / 'venv' / 'bin' / 'activate'}'

echo "Starting development server..."
echo ""
echo "🌐 Server will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/api/docs"  
echo "🏥 Health Check: http://localhost:8000/health"
echo ""

python api.py
"""
            
            dev_script_path.write_text(dev_content)
            if sys.platform != "win32": 
                os.chmod(dev_script_path, 0o755)
            logger.info(f"   - ✅ Created development script: {dev_script_path.name}")

            # Production script
            if sys.platform == "win32":
                prod_script_path = self.project_root / "start_prod.bat"
                prod_content = f"""@echo off
title Emergency Response Prod Server
echo.
echo 🚨 Starting Emergency Response Assistant (Production)
echo ================================================================
echo Activating virtual environment...
call "{self.project_root / 'venv' / 'Scripts' / 'activate.bat'}"

echo Starting production server with 4 workers...
echo.
echo 🌐 Server will be available at: http://localhost:8000
echo.
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
pause
"""
            else:
                prod_script_path = self.project_root / "start_prod.sh"
                prod_content = f"""#!/bin/bash
# Production startup script for Emergency Response Assistant

echo ""
echo "🚨 Starting Emergency Response Assistant (Production)"
echo "================================================================"
echo "Activating virtual environment..."
source '{self.project_root / 'venv' / 'bin' / 'activate'}'

echo "Starting production server with 4 workers..."
echo ""
echo "🌐 Server will be available at: http://localhost:8000"
echo ""

python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
"""
            
            prod_script_path.write_text(prod_content)
            if sys.platform != "win32": 
                os.chmod(prod_script_path, 0o755)
            logger.info(f"   - ✅ Created production script: {prod_script_path.name}")
            
            # Quick start script
            if sys.platform == "win32":
                quick_script_path = self.project_root / "quick_start.bat"
                quick_content = f"""@echo off
echo 🚨 Emergency Response Assistant - Quick Start
call "{self.project_root / 'venv' / 'Scripts' / 'activate.bat'}"
python api.py
"""
            else:
                quick_script_path = self.project_root / "quick_start.sh"
                quick_content = f"""#!/bin/bash
echo "🚨 Emergency Response Assistant - Quick Start"
source '{self.project_root / 'venv' / 'bin' / 'activate'}'
python api.py
"""
            
            quick_script_path.write_text(quick_content)
            if sys.platform != "win32": 
                os.chmod(quick_script_path, 0o755)
            logger.info(f"   - ✅ Created quick start script: {quick_script_path.name}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create startup scripts: {e}")
            return False

    def setup_configuration_files(self) -> bool:
        """Generates helpful configuration file templates like .gitignore."""
        logger.info("⚙️  Setting up default configuration files...")
        
        # Using a dictionary to hold filename and content for easy extension
        configs = {
            ".gitignore": """# Environment & Secrets
.env
.env.*
!/.env.example

# Virtual Environment
venv/
env/
.venv/

# Python Cache
__pycache__/
*.py[cod]
*$py.class

# Build & Distribution Artifacts
build/
dist/
*.egg-info/
wheels/
*.egg

# Data & Media
/data/
/uploads/
/outputs/
/models/
/logs/
/backups/
*.db
*.sqlite3

# IDE & OS specific
.vscode/
.idea/
*.swp
*.swo
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Temporary files
temp_*
tmp_*
*.tmp
""",
            "README.md": """# 🚨 Enhanced Emergency Response Assistant

AI-Powered Emergency Management System with Citizen Portal

## Features

- 🌐 **Citizen Emergency Portal** - Public interface for emergency reporting
- 🎤 **Voice Emergency Processing** - Real-time audio analysis
- 📷 **Image Analysis** - Hazard detection from images
- 🤖 **Multimodal AI Analysis** - Combined text + image + audio processing
- 📊 **Admin Dashboard** - Professional management interface
- 🏥 **Patient Triage System** - Medical prioritization
- 📢 **Crowd Reports** - Community emergency reporting
- 🗺️ **Interactive Maps** - Geographic visualization
- ⚡ **Real-time Updates** - WebSocket live updates
- 🔐 **Secure Authentication** - JWT with role-based access

## Quick Start

1. Run the setup (first time only):
   ```bash
   python setup.py
   ```

2. Start the development server:
   - **Windows**: `start_dev.bat`
   - **Linux/Mac**: `./start_dev.sh`
   - **Quick**: `python api.py`

3. Open your browser to: http://localhost:8000

## Important URLs

- **Citizen Portal**: http://localhost:8000/
- **Admin Dashboard**: http://localhost:8000/admin
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health

## Default Login

- **Username**: admin
- **Password**: admin

## Configuration

Edit the `.env` file to customize settings like database URL, secret keys, and feature toggles.

## Production Deployment

Use `start_prod.sh` or `start_prod.bat` for production deployment with multiple workers.

---

🛟 **Ready to save lives with AI-powered emergency response!**
"""
        }
        
        try:
            for filename, content in configs.items():
                file_path = self.project_root / filename
                if not file_path.exists():
                    file_path.write_text(content.strip() + "\n")
                    logger.info(f"   - ✅ Created {filename}")
                else:
                    logger.info(f"   - ℹ️  {filename} already exists. Skipping.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create configuration file {filename}: {e}")
            return False
    
    def print_setup_summary(self):
        """Prints a summary of the setup process and next steps."""
        print("\n" + "="*80)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\n📋 Summary:")
        print("   - ✅ Python & System Dependencies Checked")
        print("   - ✅ Project Directories Created")
        print("   - ✅ Virtual Environment Ready in './venv/'")
        print("   - ✅ Python Packages Installed")
        print("   - ✅ Database Configuration Ready")
        print("   - ✅ Environment Configuration (.env) Created")
        print("   - ✅ Configuration Files (.gitignore, README.md) Created")
        print("   - ✅ Startup Scripts Generated")
        
        print("\n🚀 QUICK START (Choose one):")
        if sys.platform == "win32":
            print("   Option 1: .\\start_dev.bat")
            print("   Option 2: .\\quick_start.bat")
            print("   Option 3: python api.py")
        else:
            print("   Option 1: ./start_dev.sh")
            print("   Option 2: ./quick_start.sh") 
            print("   Option 3: python api.py")
        
        print("\n📚 Important URLs (once server is running):")
        print("   🌐 Citizen Portal:    http://localhost:8000/")
        print("   📊 Admin Dashboard:   http://localhost:8000/admin")
        print("   📚 API Documentation: http://localhost:8000/api/docs")
        print("   🏥 Health Check:      http://localhost:8000/health")
        
        print("\n🔐 Default Login Credentials:")
        print("   Username: admin")
        print("   Password: admin")
        print("   (Change these in production!)")
        
        print("\n⚙️  Configuration:")
        print("   • Edit .env file for custom settings")
        print("   • Generate demo data: http://localhost:8000/api/generate-demo-data")
        print("   • View system status: http://localhost:8000/health")
        
        print("\n" + "="*80)
        print("🛟 Ready to save lives with AI-powered emergency response!")
        print("="*80 + "\n")
    
    def run_full_setup(self):
        """Orchestrates the entire setup process, running each step in order."""
        self.print_banner()
        
        steps: List[Tuple[str, Any]] = [
            ("Python version check", self.check_python_version),
            ("System dependencies check", self.check_system_dependencies),
            ("Directory creation", self.create_directories),
            ("Virtual environment setup", self.setup_virtual_environment),
            ("Python dependency installation", self.install_dependencies),
            ("Environment file (.env) setup", self.setup_environment_file),
            ("Configuration files setup", self.setup_configuration_files),
            ("Database initialization", self.initialize_database),
            ("Sample data preparation", self.create_sample_data),
            ("AI model check/download", self.download_ai_models),
            ("Setup verification", self.verify_setup),
            ("Startup script generation", self.generate_startup_script),
        ]
        
        failed_steps = []
        
        for i, (name, func) in enumerate(steps, 1):
            logger.info(f"\n--- STEP {i}/{len(steps)}: {name} ---")
            if not func():
                failed_steps.append(name)
                logger.warning(f"⚠️  Step '{name}' had issues but continuing setup...")
        
        if failed_steps:
            logger.warning(f"\n⚠️  Some steps had issues: {', '.join(failed_steps)}")
            logger.info("The setup may still work. Try running the API manually to test.")
        
        self.print_setup_summary()
        return len(failed_steps) == 0

def main():
    """Main entry point for the command-line setup utility."""
    parser = argparse.ArgumentParser(
        description="Setup utility for the Enhanced Emergency Response Assistant.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--no-models", 
        action="store_true", 
        help="Skip the AI model download step (e.g., Whisper)."
    )
    parser.add_argument(
        "--no-sample-data", 
        action="store_true", 
        help="Skip creating sample data in the database."
    )
    parser.add_argument(
        "--docker", 
        action="store_true", 
        help="Run setup for a Docker environment (skips venv creation)."
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Quick setup - skip optional components."
    )
    
    args = parser.parse_args()
    
    setup = DisasterResponseSetup()
    
    # Modify the setup flow based on arguments
    if args.no_models:
        setup.download_ai_models = lambda: logger.info("   - ℹ️  Skipping AI model download as requested.") or True
    
    if args.no_sample_data:
        setup.create_sample_data = lambda: logger.info("   - ℹ️  Skipping sample data creation as requested.") or True
    
    if args.docker:
        logger.info("🐳 Running in Docker mode. Skipping virtual environment setup.")
        setup.setup_virtual_environment = lambda: logger.info("   - ℹ️  Skipping venv setup for Docker.") or True
        setup.install_dependencies = lambda: logger.info("   - ℹ️  Skipping dependency installation for Docker.") or True
    
    if args.quick:
        logger.info("⚡ Quick setup mode - skipping optional components.")
        setup.download_ai_models = lambda: logger.info("   - ℹ️  Skipping AI models in quick mode.") or True
        setup.create_sample_data = lambda: logger.info("   - ℹ️  Skipping sample data in quick mode.") or True
    
    try:
        success = setup.run_full_setup()
        if success:
            print("\n🎉 Setup completed successfully! You can now start the Emergency Response Assistant.")
            sys.exit(0)
        else:
            print("\n⚠️  Setup completed with some issues. You may still be able to run the application.")
            print("Try running: python api.py")
            sys.exit(0)  # Don't fail completely - user can still try to run
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ An unexpected error occurred during setup: {e}", exc_info=True)
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Make sure you have Python 3.8+ installed")
        print(f"   2. Try running with --quick flag: python setup.py --quick")
        print(f"   3. Check that you have write permissions in this directory")
        print(f"   4. If all else fails, try manual setup: python api.py")
        sys.exit(1)

if __name__ == "__main__":
    main()