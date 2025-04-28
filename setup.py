import os
import sys
import subprocess

def setup_environment():
    """Configura l'ambiente di lavoro per il classificatore di rocce"""
    print("Configurazione dell'ambiente per il classificatore di rocce...")
    
    # Verifica che la cartella pacchetti_offline esista
    if not os.path.exists("pacchetti_offline"):
        print("AVVISO: La cartella 'pacchetti_offline' non esiste!")
        
        # Verifica se esiste il file zip
        if os.path.exists("pacchetti_offline.zip"):
            print("Trovato 'pacchetti_offline.zip'. Estrazione in corso...")
            try:
                subprocess.run([sys.executable, "scarica_pacchetti.py", "estrai"], check=True)
            except:
                print("ERRORE: Impossibile estrarre pacchetti_offline.zip.")
                print("Prova a estrarre manualmente il file zip nella cartella 'pacchetti_offline'.")
                return
        else:
            print("ERRORE: Pacchetti offline non trovati!")
            print("Per completare l'installazione offline, è necessario:")
            print("1. Scaricare il file 'pacchetti_offline.zip' dalla pagina delle release del progetto")
            print("2. Posizionare il file nella directory principale del progetto")
            print("3. Eseguire nuovamente questo script")
            print("\nAlternativamente, se hai connessione internet, puoi scaricare i pacchetti automaticamente")
            print("eseguendo: py scarica_pacchetti.py")
            return
    
    # Crea l'ambiente virtuale se non esiste
    if not os.path.exists("rocce_env"):
        print("Creazione dell'ambiente virtuale...")
        subprocess.run([sys.executable, "-m", "venv", "rocce_env"])
    
    # Attiva l'ambiente virtuale e installa le dipendenze
    if sys.platform == "win32":
        pip_path = os.path.join("rocce_env", "Scripts", "pip")
    else:
        pip_path = os.path.join("rocce_env", "bin", "pip")
    
    print("Installazione delle dipendenze dai pacchetti locali...")
    subprocess.run([pip_path, "install", "--upgrade", "pip", "--no-index", "--find-links=pacchetti_offline"])
    subprocess.run([pip_path, "install", "-r", "requirements.txt", "--no-index", "--find-links=pacchetti_offline"])
    
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
    print("    py classificatore_rocce.py")
    print("\nAlternativa più semplice (solo Windows):")
    print("    Doppio click su avvia_classificatore.bat")

if __name__ == "__main__":
    setup_environment()