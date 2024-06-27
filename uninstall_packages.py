import subprocess
import sys

def get_installed_packages():
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return result.stdout.splitlines()

def uninstall_packages(packages):
    for package in packages:
        if not package.startswith("pip") and not package.startswith("setuptools") and not package.startswith("wheel"):
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package])

def reinstall_base_packages():
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

if __name__ == "__main__":
    packages = get_installed_packages()
    uninstall_packages(packages)
    reinstall_base_packages()
    print("All packages have been uninstalled and base packages reinstalled.")