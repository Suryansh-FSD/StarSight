import os
import sys
import subprocess
import getpass
from pathlib import Path

def bootstrap_repo(project_root: Path) -> None:
    """
    Validates, authenticates, and synchronizes the private repository on Google Drive.
    Performs 'git pull' if the repo is already present, or handles credentials prompting
    and clones it if missing.
    
    Args:
        project_root: Path where the repository resides in Google Drive.
    """
    print(f"Bootstrapping StarSight repository at: {project_root.resolve()}")
    
    # Ensure project root parent folder exists
    project_root.parent.mkdir(parents=True, exist_ok=True)
    
    git_dir = project_root / ".git"
    
    if git_dir.exists():
        print("Repository detected on Google Drive. Syncing latest changes (git pull)...")
        try:
            # Execute git pull inside the repository directory
            result = subprocess.run(
                ["git", "pull"],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            print(result.stdout)
            print("Repository synchronized successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: 'git pull' encountered an issue: {e.stderr.strip()}")
            print("Proceeding with existing files.")
    else:
        print("StarSight repository not found in Google Drive. Commencing authenticated clone...")
        username = input("GitHub Username: ")
        # Secure password input (avoids print exposure)
        token = getpass.getpass("GitHub Personal Access Token (PAT): ")
        
        if not username or not token:
            raise ValueError("Username and Personal Access Token (PAT) are mandatory to clone this private repository.")
            
        authenticated_url = f"https://{username}:{token}@github.com/Suryansh-FSD/StarSight.git"
        
        print(f"Cloning private repository into: {project_root.resolve()}")
        try:
            subprocess.run(
                ["git", "clone", authenticated_url, str(project_root)],
                check=True
            )
            print("Repository cloned successfully to Google Drive.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone private repository: Check username and PAT validity.")
            
    # Add project root to sys.path if not present
    src_path = str(project_root.resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        
    # Validate that imports from src work successfully
    try:
        from src.utils.colab import print_environment_summary
        print("Verification: 'src' modules imported and validated successfully.")
    except ImportError as e:
        raise ImportError(f"Validation failed: Unable to import 'src' modules from project root. Details: {e}")
