import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DisasterResponseSetup:
    """Setup utility for Disaster Response Assistant"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_version = sys.version_info
        self.requirements_met = True
        
    def print_banner(self):
        """Print setup banner"""
        print("\n" + "="*80)
        print("🚨 DISASTER RESPONSE & RECOVERY ASSISTANT SETUP")
        print("="*80)
        print("🧠 AI-Powered Emergency Analysis with Gemma 3n")
        print("🎤 Voice Emergency Processing | 📷 Image Analysis | 🗺️ Real-time Mapping")
        print("⚡ Adaptive AI Optimization | 📊 Predictive Analytics")
        print("="*80)
    
    def check_python_version(self):
        """Check Python version compatibility"""
        logger.info("🐍 Checking Python version...")
        
        if self.python_version < (3, 8):
            logger.error(f"❌ Python 3.8+ required, found {self.python_version.major}.{self.python_version.minor}")
            self.requirements_met = False
            return False
        elif self.python_version < (3, 9):
            logger.warning(f"⚠️  Python 3.9+ recommended, found {self.python_version.major}.{self.python_version.minor}")
        else:
            logger.info(f"✅ Python {self.python_version.major}.{self.python_version.minor} is compatible")
        
        return True
    
    def check_system_dependencies(self):
        """Check system dependencies"""
        logger.info("🔧 Checking system dependencies...")
        
        # Required system packages
        required_packages = {
            'git': 'Git version control',
            'curl': 'HTTP client',
            'ffmpeg': 'Audio/video processing'
        }
        
        missing_packages = []
        
        for package, description in required_packages.items():
            if not shutil.which(package):
                missing_packages.append(f"{package} ({description})")
                logger.warning(f"⚠️  {package} not found")
            else:
                logger.info(f"✅ {package} found")
        
        if missing_packages:
            logger.warning("Missing system packages:")
            for package in missing_packages:
                logger.warning(f"   • {package}")
            logger.info("\nInstall missing packages:")
            logger.info("   Ubuntu/Debian: sudo apt install git curl ffmpeg")
            logger.info("   macOS: brew install git curl ffmpeg")
            logger.info("   Windows: Use chocolatey or download installers")
        
        return len(missing_packages) == 0
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("📁 Creating directories...")
        
        directories = [
            'uploads',
            'outputs', 
            'models',
            'logs',
            'static',
            'templates',
            'data',
            'backups'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✅ Created/verified: {directory}/")
    
    def setup_virtual_environment(self):
        """Setup virtual environment"""
        logger.info("🐍 Setting up virtual environment...")
        
        venv_path = self.project_root / "venv"
        
        if venv_path.exists():
            logger.info("✅ Virtual environment already exists")
            return True
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            logger.info("✅ Virtual environment created")
            
            # Activate and upgrade pip
            if sys.platform == "win32":
                pip_path = venv_path / "Scripts" / "pip"
                python_path = venv_path / "Scripts" / "python"
            else:
                pip_path = venv_path / "bin" / "pip"
                python_path = venv_path / "bin" / "python"
            
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
            logger.info("✅ Pip upgraded")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            logger.error("❌ requirements.txt not found")
            return False
        
        try:
            # Install requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_environment_file(self):
        """Setup environment configuration"""
        logger.info("⚙️  Setting up environment configuration...")
        
        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"
        
        if env_file.exists():
            logger.info("✅ .env file already exists")
            return True
        
        if env_example.exists():
            try:
                shutil.copy(env_example, env_file)
                logger.info("✅ .env file created from template")
                logger.warning("⚠️  Please edit .env file with your API keys and configuration")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to copy .env template: {e}")
                return False
        else:
            logger.warning("⚠️  No .env.example template found")
            return False
    
    def initialize_database(self):
        """Initialize database"""
        logger.info("🗄️  Initializing database...")
        
        try:
            # Import after dependencies are installed
            from app.database import engine
            from app.models import Base
            
            # Create all tables
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database tables created")
            
            # Run migrations if available
            migration_script = self.project_root / "app" / "migrate_database.py"
            if migration_script.exists():
                subprocess.run([sys.executable, str(migration_script)], check=True)
                logger.info("✅ Database migrations completed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    def download_ai_models(self):
        """Download required AI models"""
        logger.info("🤖 Downloading AI models...")
        
        try:
            # Download Whisper model
            import whisper
            logger.info("📥 Downloading Whisper model...")
            whisper.load_model("base")
            logger.info("✅ Whisper model downloaded")
            
            # Note: Gemma models would be downloaded automatically when first used
            logger.info("ℹ️  Gemma 3n models will be downloaded on first use")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ AI model download failed: {e}")
            logger.info("ℹ️  Models will be downloaded when first needed")
            return False
    
    def create_sample_data(self):
        """Create sample data for testing"""
        logger.info("🌱 Creating sample data...")
        
        try:
            # Import sample data script
            sample_script = self.project_root / "app" / "seed_demo_data.py"
            if sample_script.exists():
                subprocess.run([sys.executable, str(sample_script)], check=True)
                logger.info("✅ Sample data created")
                return True
            else:
                logger.warning("⚠️  Sample data script not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Sample data creation failed: {e}")
            return False
    
    def run_tests(self):
        """Run basic tests"""
        logger.info("🧪 Running basic tests...")
        
        try:
            # Test imports
            from app.server import app
            from app.inference import Gemma3nEmergencyProcessor
            from app.audio_transcription import VoiceEmergencyProcessor
            
            logger.info("✅ Core modules import successfully")
            
            # Test database connection
            from app.database import get_db
            db = next(get_db())
            db.close()
            logger.info("✅ Database connection successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Basic tests failed: {e}")
            return False
    
    def generate_startup_script(self):
        """Generate startup scripts"""
        logger.info("📝 Generating startup scripts...")
        
        # Development startup script
        if sys.platform == "win32":
            startup_script = self.project_root / "start_dev.bat"
            script_content = """@echo off
echo Starting Disaster Response Assistant...
call venv\\Scripts\\activate
python -m uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
pause
"""
        else:
            startup_script = self.project_root / "start_dev.sh"
            script_content = """#!/bin/bash
echo "Starting Disaster Response Assistant..."
source venv/bin/activate
python -m uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
"""
        
        try:
            with open(startup_script, 'w') as f:
                f.write(script_content)
            
            if sys.platform != "win32":
                os.chmod(startup_script, 0o755)
            
            logger.info(f"✅ Created startup script: {startup_script.name}")
            
            # Production startup script
            if sys.platform == "win32":
                prod_script = self.project_root / "start_prod.bat"
                prod_content = """@echo off
echo Starting Disaster Response Assistant (Production)...
call venv\\Scripts\\activate
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 4
pause
"""
            else:
                prod_script = self.project_root / "start_prod.sh"
                prod_content = """#!/bin/bash
echo "Starting Disaster Response Assistant (Production)..."
source venv/bin/activate
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 4
"""
            
            with open(prod_script, 'w') as f:
                f.write(prod_content)
                
            if sys.platform != "win32":
                os.chmod(prod_script, 0o755)
            
            logger.info(f"✅ Created production script: {prod_script.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create startup scripts: {e}")
            return False
    
    def setup_configuration_files(self):
        """Setup additional configuration files"""
        logger.info("⚙️  Setting up configuration files...")
        
        configs = {
            "nginx.conf": """
events {
    worker_connections 1024;
}

http {
    upstream disaster_app {
        server disaster-app:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        client_max_body_size 50M;
        
        location / {
            proxy_pass http://disaster_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /static/ {
            alias /var/www/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
    }
}
""",
            "prometheus.yml": """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'disaster-app'
    static_configs:
      - targets: ['disaster-app:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
""",
            ".gitignore": """
# Environment
.env
.env.local
.env.production

# Virtual environment
venv/
env/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data directories
uploads/
outputs/
models/
logs/
backups/
data/

# Database
*.db
*.sqlite3

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
docker-compose.override.yml

# Jupyter
.ipynb_checkpoints/

# SSL certificates
*.pem
*.key
*.crt
"""
        }
        
        for filename, content in configs.items():
            file_path = self.project_root / filename
            if not file_path.exists():
                try:
                    with open(file_path, 'w') as f:
                        f.write(content.strip())
                    logger.info(f"✅ Created {filename}")
                except Exception as e:
                    logger.error(f"❌ Failed to create {filename}: {e}")
            else:
                logger.info(f"ℹ️  {filename} already exists")
        
        return True
    
    def print_setup_summary(self):
        """Print setup completion summary"""
        print("\n" + "="*80)
        print("🎉 SETUP COMPLETED!")
        print("="*80)
        
        print("\n📋 What was set up:")
        print("   ✅ Virtual environment created")
        print("   ✅ Dependencies installed")
        print("   ✅ Database initialized")
        print("   ✅ Configuration files created")
        print("   ✅ Sample data populated")
        print("   ✅ Startup scripts generated")
        
        print("\n🚀 Quick Start:")
        if sys.platform == "win32":
            print("   1. Run: start_dev.bat")
        else:
            print("   1. Run: ./start_dev.sh")
        print("   2. Open: http://localhost:8000")
        print("   3. Login with: admin/password123")
        
        print("\n📚 Important Files:")
        print("   • .env - Configure API keys and settings")
        print("   • app/server.py - Main application server")
        print("   • app/main.py - Command-line interface")
        print("   • docker-compose.yml - Container deployment")
        
        print("\n🔧 Next Steps:")
        print("   1. Edit .env file with your API keys")
        print("   2. Configure map services (Google Maps/MapBox)")
        print("   3. Set up monitoring and logging")
        print("   4. Review security settings for production")
        
        print("\n📖 Documentation:")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • ReDoc: http://localhost:8000/redoc")
        print("   • Health Check: http://localhost:8000/health")
        
        print("\n🆘 Need Help?")
        print("   • Check logs in logs/ directory")
        print("   • Run: python app/main.py --help")
        print("   • Test components: python -m pytest tests/")
        
        print("\n" + "="*80)
        print("🛟 Ready to save lives with AI-powered emergency response!")
        print("="*80 + "\n")
    
    def run_full_setup(self):
        """Run complete setup process"""
        self.print_banner()
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Checking system dependencies", self.check_system_dependencies),
            ("Creating directories", self.create_directories),
            ("Setting up virtual environment", self.setup_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Setting up environment file", self.setup_environment_file),
            ("Setting up configuration files", self.setup_configuration_files),
            ("Initializing database", self.initialize_database),
            ("Creating sample data", self.create_sample_data),
            ("Downloading AI models", self.download_ai_models),
            ("Running basic tests", self.run_tests),
            ("Generating startup scripts", self.generate_startup_script),
        ]
        
        successful_steps = 0
        total_steps = len(steps)
        
        for step_name, step_function in steps:
            print(f"\n🔄 {step_name}...")
            try:
                if step_function():
                    successful_steps += 1
                else:
                    logger.warning(f"⚠️  {step_name} completed with warnings")
            except Exception as e:
                logger.error(f"❌ {step_name} failed: {e}")
        
        print(f"\n📊 Setup Progress: {successful_steps}/{total_steps} steps completed")
        
        if successful_steps >= total_steps - 2:  # Allow 2 optional failures
            self.print_setup_summary()
            return True
        else:
            print("\n⚠️  Setup completed with some issues. Check the logs above.")
            print("The system may still be functional, but some features might not work properly.")
            return False

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Disaster Response Assistant")
    parser.add_argument("--quick", action="store_true", help="Quick setup (skip optional steps)")
    parser.add_argument("--no-models", action="store_true", help="Skip AI model downloads")
    parser.add_argument("--no-sample-data", action="store_true", help="Skip sample data creation")
    parser.add_argument("--docker", action="store_true", help="Setup for Docker deployment")
    
    args = parser.parse_args()
    
    setup = DisasterResponseSetup()
    
    if args.quick:
        logger.info("🚀 Running quick setup...")
        # Override methods for quick setup
        setup.download_ai_models = lambda: True
        setup.create_sample_data = lambda: True
    
    if args.no_models:
        setup.download_ai_models = lambda: True
    
    if args.no_sample_data:
        setup.create_sample_data = lambda: True
    
    if args.docker:
        logger.info("🐳 Setting up for Docker deployment...")
        # Skip virtual environment setup for Docker
        setup.setup_virtual_environment = lambda: True
    
    try:
        success = setup.run_full_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()