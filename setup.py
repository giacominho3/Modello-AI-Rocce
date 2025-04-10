import os
import sys
import subprocess

def setup_environment():
    """Configura l'ambiente di lavoro per il classificatore di rocce"""
    print("Configurazione dell'ambiente per il classificatore di rocce...")
    
    # Crea l'ambiente virtuale se non esiste
    if not os.path.exists("rocce_env"):
        print("Creazione dell'ambiente virtuale...")
        subprocess.run([sys.executable, "-m", "venv", "rocce_env"])
    
    # Attiva l'ambiente virtuale e installa le dipendenze
    if sys.platform == "win32":
        pip_path = os.path.join("rocce_env", "Scripts", "pip")
    else:
        pip_path = os.path.join("rocce_env", "bin", "pip")
    
    print("Installazione delle dipendenze...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"])
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    
    # Crea le cartelle del dataset se non esistono
    os.makedirs(os.path.join("dataset_rocce", "sedimentarie"), exist_ok=True)
    os.makedirs(os.path.join("dataset_rocce", "ignee"), exist_ok=True)
    os.makedirs(os.path.join("dataset_rocce", "metamorfiche"), exist_ok=True)
    
    print("\nConfigurazione completata!")
    print("\nPer attivare l'ambiente virtuale:")
    if sys.platform == "win32":
        print("    rocce_env\\Scripts\\activate")
    else:
        print("    source rocce_env/bin/activate")
    
    print("\nPer eseguire il classificatore:")
    print("    python classificatore_rocce.py")

if __name__ == "__main__":
    setup_environment()