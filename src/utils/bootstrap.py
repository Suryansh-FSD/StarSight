import sys
import subprocess
import getpass
from pathlib import Path

def download_default_data(raw_dir: Path) -> None:
    """
    Downloads Kepler long-cadence FITS files for the default target list from the MAST archive.
    """
    print(f"Checking for raw FITS datasets in: {raw_dir.resolve()}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    targets = [
        ("Kepler-10", 3),
        ("Kepler-22", 0),
        ("Kepler-90", 1),
        ("Kepler-186", 3),
        ("Kepler-452", 3),
        ("Kepler-62", 3),
        ("Kepler-7", 3),
        ("Kepler-8", 0)
    ]
    
    # We load lightkurve locally
    try:
        import lightkurve as lk
    except ImportError:
        print("Warning: lightkurve not installed yet. Skipping download step.")
        return
        
    for target, quarter in targets:
        clean_target = target.replace(' ', '-').lower()
        # Check if file exists already
        matches = list(raw_dir.glob(f"{clean_target}_*.fits"))
        if matches:
            print(f"Dataset already exists for {target}. Skipping.")
            continue
            
        print(f"Downloading light curves for {target} (Quarter {quarter})...")
        try:
            search_result = lk.search_lightcurve(target, author="Kepler", quarter=quarter, cadence="long")
            if len(search_result) > 0:
                lc = search_result[0].download()
                if lc is not None:
                    filename = f"{clean_target}_q{quarter}.fits"
                    filepath = raw_dir / filename
                    lc.to_fits(str(filepath), overwrite=True)
                    print(f"Successfully downloaded and saved: {filename}")
            else:
                print(f"Warning: No light curve found for {target} in Quarter {quarter}")
        except Exception as e:
            print(f"Error downloading {target}: {e}")

def bootstrap_repo(project_root: Path) -> None:
    """
    Validates, authenticates, and synchronizes the private repository on Google Drive.
    Performs 'git pull' if the repo is already present, or handles credentials prompting
    and clones/initializes it if missing.
    
    Args:
        project_root: Path where the repository resides in Google Drive.
    """
    print(f"Bootstrapping StarSight repository at: {project_root.resolve()}")
    
    # Ensure project root parent folder exists
    project_root.parent.mkdir(parents=True, exist_ok=True)
    
    # Attempt automatic Google Drive mount if in Colab
    if 'google.colab' in sys.modules:
        try:
            from google.colab import drive
            print("Google Colab detected. Attempting to mount Google Drive...")
            drive.mount('/content/drive')
        except Exception as e:
            print(f"Warning: Google Drive mount failed: {e}")
            
    git_dir = project_root / ".git"
    
    if git_dir.exists():
        print("Repository detected on Google Drive. Syncing latest changes (git pull)...")
        try:
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
            print(f"Warning: 'git pull' failed: {e.stderr.strip()}")
            print("Attempting to re-authenticate and pull...")
            try:
                username = input("GitHub Username: ")
                token = getpass.getpass("GitHub Personal Access Token (PAT): ")
                if username and token:
                    authenticated_url = f"https://{username}:{token}@github.com/Suryansh-FSD/StarSight.git"
                    subprocess.run(["git", "remote", "set-url", "origin", authenticated_url], cwd=str(project_root), check=True)
                    subprocess.run(["git", "pull"], cwd=str(project_root), check=True)
                    print("Re-authentication and pull successful.")
                else:
                    print("Skipping re-authentication. Using existing local files.")
            except Exception as re_err:
                print(f"Re-authentication failed: {re_err}. Proceeding with existing files.")
    else:
        # Check if we are running in Colab and cloning is needed
        if 'google.colab' in sys.modules:
            print("StarSight repository not found or not initialized in Google Drive.")
            username = input("GitHub Username: ")
            token = getpass.getpass("GitHub Personal Access Token (PAT): ")
            
            if not username or not token:
                raise ValueError("Username and Personal Access Token (PAT) are mandatory to clone this private repository.")
                
            authenticated_url = f"https://{username}:{token}@github.com/Suryansh-FSD/StarSight.git"
            
            if project_root.exists() and any(project_root.iterdir()):
                print("Target directory exists and is not empty. Initializing git in-place...")
                try:
                    subprocess.run(["git", "init"], cwd=str(project_root), check=True)
                    subprocess.run(["git", "remote", "remove", "origin"], cwd=str(project_root), stderr=subprocess.DEVNULL)
                    subprocess.run(["git", "remote", "add", "origin", authenticated_url], cwd=str(project_root), check=True)
                    subprocess.run(["git", "fetch", "origin"], cwd=str(project_root), check=True)
                    subprocess.run(["git", "checkout", "-B", "main"], cwd=str(project_root), check=True)
                    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=str(project_root), check=True)
                    subprocess.run(["git", "branch", "--set-upstream-to=origin/main", "main"], cwd=str(project_root), check=True)
                    print("Repository initialized and checked out successfully in-place.")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Failed to initialize repository in-place: Check username and PAT.")
            else:
                print(f"Cloning private repository into empty directory: {project_root.resolve()}")
                try:
                    subprocess.run(["git", "clone", authenticated_url, str(project_root)], check=True)
                    print("Repository cloned successfully to Google Drive.")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Failed to clone private repository: Check username and PAT validity.")
                    
    # Install dependencies from requirements.txt if running in Google Colab
    if 'google.colab' in sys.modules:
        requirements_file = project_root / "requirements.txt"
        if requirements_file.exists():
            print("Google Colab detected. Installing project dependencies from requirements.txt...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
                    check=True
                )
                print("All dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Package installation via pip failed: {e}")
        else:
            print("Warning: requirements.txt not found. Skipping dependency installation.")

    # Add project root to sys.path if not present
    src_path = str(project_root.resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        
    # Download default light curve data if raw directory is empty
    raw_dir = project_root / "data" / "raw"
    download_default_data(raw_dir)
        
    # Validate that imports from src work successfully
    try:
        from src.utils.colab import print_environment_summary
        print("Verification: 'src' modules imported and validated successfully.")
    except ImportError as e:
        raise ImportError(f"Validation failed: Unable to import 'src' modules from project root. Details: {e}")
