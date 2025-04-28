import os
import sys
import subprocess
import zipfile
import shutil

def scarica_pacchetti():
    """
    Script per scaricare i pacchetti necessari per l'installazione offline.
    Questo script dovrebbe essere eseguito su un computer con accesso a internet.
    """
    print("Scaricamento pacchetti per installazione offline...")
    
    # Verifica se pip è disponibile
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True)
    except:
        print("ERRORE: pip non è installato o non è disponibile.")
        return False
    
    # Crea la directory pacchetti_offline se non esiste
    if not os.path.exists("pacchetti_offline"):
        os.makedirs("pacchetti_offline")
    
    # Scarica i pacchetti
    print("Scaricamento pacchetti in corso (questo potrebbe richiedere alcuni minuti)...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "download",
            "-r", "requirements.txt",
            "-d", "./pacchetti_offline"
        ], check=True)
        print("Scaricamento completato con successo!")
    except subprocess.CalledProcessError:
        print("ERRORE: Impossibile scaricare i pacchetti. Verifica la tua connessione internet.")
        return False
    
    # Crea un archivio ZIP dei pacchetti
    print("Creazione archivio pacchetti_offline.zip...")
    try:
        with zipfile.ZipFile('pacchetti_offline.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk('pacchetti_offline'):
                for file in files:
                    zipf.write(
                        os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file), 'pacchetti_offline')
                    )
        print(f"Archivio creato: {os.path.abspath('pacchetti_offline.zip')}")
    except Exception as e:
        print(f"ERRORE durante la creazione dell'archivio: {e}")
        return False
    
    return True

def estrai_pacchetti():
    """
    Estrae i pacchetti dall'archivio ZIP nella cartella pacchetti_offline.
    Questo script dovrebbe essere eseguito dopo aver scaricato l'archivio.
    """
    print("Estrazione pacchetti offline...")
    
    # Verifica se l'archivio esiste
    if not os.path.exists("pacchetti_offline.zip"):
        print("ERRORE: Il file 'pacchetti_offline.zip' non esiste.")
        print("Scarica il file dal repository delle release e posizionalo nella directory principale.")
        return False
    
    # Crea la directory pacchetti_offline se non esiste
    if os.path.exists("pacchetti_offline"):
        print("Rimozione directory pacchetti_offline esistente...")
        shutil.rmtree("pacchetti_offline")
    
    os.makedirs("pacchetti_offline")
    
    # Estrai l'archivio
    print("Estrazione archivio in corso...")
    try:
        with zipfile.ZipFile('pacchetti_offline.zip', 'r') as zip_ref:
            zip_ref.extractall("pacchetti_offline")
        print("Estrazione completata con successo!")
    except Exception as e:
        print(f"ERRORE durante l'estrazione: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "estrai":
        estrai_pacchetti()
    else:
        scarica_pacchetti()