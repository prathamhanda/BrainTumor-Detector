"""
Production Deployment Checklist for Brain Tumor AI Classifier
Run this script to verify your setup before deployment
"""

import os
import sys
from pathlib import Path
import subprocess

def check_git_status():
    """Check git repository status"""
    print("🔍 Checking Git Status...")
    try:
        # Check if we're in a git repo
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            if result.stdout.strip():
                print("⚠️ Uncommitted changes found:")
                print(result.stdout)
                return False
            else:
                print("✅ Git repository is clean")
                return True
        else:
            print("❌ Not a git repository or git not found")
            return False
    except Exception as e:
        print(f"❌ Git check failed: {e}")
        return False

def check_required_files():
    """Check if all required files exist"""
    print("📁 Checking Required Files...")
    
    required_files = [
        "app.py",
        "requirements.txt", 
        "download_models.py",
        "src/model.py",
        "src/utils.py", 
        "src/segmentation.py",
        "sample/s1.JPG",
        "sample/s2.JPG", 
        "sample/s3.JPG",
        "README.md",
        "CLOUD_SETUP.md",
        ".gitignore"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_gitignore():
    """Check if .gitignore properly excludes models"""
    print("🚫 Checking .gitignore...")
    
    try:
        with open('.gitignore', 'r') as f:
            content = f.read()
        
        required_entries = ['models/', '*.pt', '__pycache__/', '.streamlit/']
        missing_entries = []
        
        for entry in required_entries:
            if entry not in content:
                missing_entries.append(entry)
        
        if missing_entries:
            print(f"⚠️ Missing .gitignore entries: {missing_entries}")
            return False
        else:
            print("✅ .gitignore properly configured")
            return True
            
    except Exception as e:
        print(f"❌ .gitignore check failed: {e}")
        return False

def check_download_urls():
    """Check if download URLs are configured"""
    print("🔗 Checking Download URLs...")
    
    try:
        with open('download_models.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for placeholder URLs
        if "YOUR_" in content:
            print("⚠️ Placeholder URLs found in download_models.py")
            print("📋 Action required: Update model URLs in download_models.py")
            print("📖 See CLOUD_SETUP.md for instructions")
            return False
        
        # Check for actual URLs (Google Drive, Dropbox, etc.)
        if ("drive.google.com" in content or 
            "dropbox.com" in content or 
            "huggingface.co" in content or
            "amazonaws.com" in content):
            print("✅ Download URLs configured with cloud storage")
            return True
        else:
            print("⚠️ No recognizable cloud storage URLs found")
            return False
            
    except Exception as e:
        print(f"❌ Download URL check failed: {e}")
        return False

def check_streamlit_config():
    """Check Streamlit configuration"""
    print("🎛️ Checking Streamlit Configuration...")
    
    # Check if secrets.toml exists (optional)
    if os.path.exists('.streamlit/secrets.toml'):
        print("✅ Streamlit secrets.toml found")
    else:
        print("💡 No secrets.toml found (optional)")
    
    # Check app.py for proper configuration
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_imports = ['streamlit', 'torch', 'PIL', 'download_models']
        missing_imports = []
        
        for imp in required_imports:
            if imp not in content:
                missing_imports.append(imp)
        
        if missing_imports:
            print(f"❌ Missing imports in app.py: {missing_imports}")
            return False
        else:
            print("✅ Streamlit app properly configured")
            return True
            
    except Exception as e:
        print(f"❌ Streamlit config check failed: {e}")
        return False

def create_deployment_summary():
    """Create deployment summary"""
    print("\n" + "=" * 60)
    print("🚀 DEPLOYMENT SUMMARY")
    print("=" * 60)
    
    print("\n📦 What's Ready for Deployment:")
    print("✅ Streamlit app with integrated model downloader")
    print("✅ Error handling for missing models") 
    print("✅ Demo mode for users without models")
    print("✅ Professional UI with progress indicators")
    print("✅ Complete documentation and setup guides")
    
    print("\n🔧 For Full AI Functionality:")
    print("1. Upload your model files to cloud storage")
    print("2. Update URLs in download_models.py") 
    print("3. Test the download feature locally")
    print("4. Deploy to Streamlit Cloud")
    
    print("\n🌐 Deployment Options:")
    print("• Streamlit Cloud (recommended)")
    print("• Heroku")
    print("• AWS/GCP/Azure")
    print("• Self-hosted")
    
    print("\n📖 Documentation:")
    print("• README.md - Main project documentation")
    print("• CLOUD_SETUP.md - Cloud storage configuration")
    print("• test_ai_features.py - Local testing script")

def main():
    """Run all deployment checks"""
    print("🧠 Brain Tumor AI Classifier - Deployment Checker")
    print("=" * 60)
    
    checks = [
        ("Git Status", check_git_status),
        ("Required Files", check_required_files), 
        ("GitIgnore Config", check_gitignore),
        ("Download URLs", check_download_urls),
        ("Streamlit Config", check_streamlit_config)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 Ready for deployment!")
    elif passed_checks >= total_checks - 1:
        print("⚠️ Almost ready - check warnings above")
    else:
        print("❌ Issues found - please fix before deployment")
    
    create_deployment_summary()
    
    print(f"\n💡 Next Steps:")
    if passed_checks == total_checks:
        print("1. Push your changes to GitHub")
        print("2. Configure cloud storage URLs if needed")  
        print("3. Deploy to Streamlit Cloud")
        print("4. Test the live deployment")
    else:
        print("1. Fix the issues mentioned above")
        print("2. Run this script again")
        print("3. Proceed with deployment when all checks pass")

if __name__ == "__main__":
    main()
