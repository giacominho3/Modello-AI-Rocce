import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import (
    crea_dataset, crea_modello_semplice, addestra_modello_base,
    visualizza_risultati_base, salva_modello_base, carica_modello_base,
    predici_immagine_base
)

def main():
    print("=" * 70)
    print("CLASSIFICATORE DI ROCCE CON INTELLIGENZA ARTIFICIALE")
    print("=" * 70)
    
    # Verifica se il modello esiste già
    model_path = "modello_rocce_base.keras"
    model_exists = os.path.exists(model_path) or os.path.exists(model_path.replace('.keras', '.h5'))
    
    # Verifica se la cartella del dataset esiste e contiene sottocartelle
    dataset_folder = "dataset_rocce"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder, exist_ok=True)
        os.makedirs(os.path.join(dataset_folder, "sedimentarie"), exist_ok=True)
        os.makedirs(os.path.join(dataset_folder, "ignee"), exist_ok=True)
        os.makedirs(os.path.join(dataset_folder, "metamorfiche"), exist_ok=True)
        print(f"\nCreata struttura delle cartelle in: {dataset_folder}")
        print("Aggiungi immagini nelle rispettive sottocartelle prima di addestrare il modello.")
        
    subdirs = [f.name for f in os.scandir(dataset_folder) if f.is_dir()]
    dataset_ready = len(subdirs) >= 2
    
    # Modalità di utilizzo
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        # Modalità interattiva
        print("\nScegli un'operazione:")
        print("1. Addestra un nuovo modello")
        print("2. Classifica un'immagine")
        print("3. Visualizza informazioni sul modello")
        print("4. Esci")
        
        choice = input("\nInserisci il numero dell'operazione: ")
        
        if choice == "1":
            mode = "train"
        elif choice == "2":
            mode = "predict"
        elif choice == "3":
            mode = "info"
        else:
            print("Uscita dal programma.")
            return
    
    # Addestramento del modello
    if mode == "train":
        if not dataset_ready:
            print(f"\nERRORE: Il dataset non è pronto per l'addestramento.")
            print(f"Inserisci immagini nelle sottocartelle di {dataset_folder}")
            print("Servono almeno due classi con immagini.")
            return
            
        print("\nAvvio addestramento del modello...")
        
        try:
            # Crea il dataset
            train_ds, val_ds, class_names = crea_dataset(dataset_folder)
            
            # Crea e addestra il modello
            model = crea_modello_semplice(len(class_names))
            history = addestra_modello_base(model, train_ds, val_ds, epoche=15)
            
            # Visualizza e salva i risultati
            visualizza_risultati_base(history)
            salva_modello_base(model)
            
            print("\nAddestramento completato con successo!")
            print(f"Il modello può classificare queste classi: {class_names}")
            
        except Exception as e:
            print(f"\nErrore durante l'addestramento: {e}")
    
    # Predizione su nuova immagine
    elif mode == "predict":
        if not model_exists:
            print("\nERRORE: Nessun modello trovato. Addestra prima un modello.")
            return
            
        image_path = ""
        if len(sys.argv) > 2:
            image_path = sys.argv[2]
        else:
            image_path = input("\nInserisci il percorso dell'immagine da classificare: ")
        
        if not os.path.exists(image_path):
            print(f"\nERRORE: L'immagine {image_path} non esiste.")
            return
            
        try:
            # Carica il modello e fai la predizione
            model = carica_modello_base()
            
            # Determina le classi in base alle sottocartelle del dataset
            if dataset_ready:
                class_names = subdirs
            else:
                class_names = ["sedimentarie", "ignee", "metamorfiche"]
                
            risultato = predici_immagine_base(model, image_path, class_names)
            
            if risultato:
                print("\nRisultato della classificazione:")
                print(f"Tipo di roccia: {risultato['class']}")
                print(f"Confidenza: {risultato['confidence']:.2f}")
                
                # Mostra anche altre probabilità
                print("\nProbabilità per tutte le classi:")
                for i, cls in enumerate(class_names):
                    print(f"- {cls}: {risultato['all_predictions'][i]:.4f}")
            
        except Exception as e:
            print(f"\nErrore durante la predizione: {e}")
    
    # Informazioni sul modello
    elif mode == "info":
        print("\nInformazioni sul classificatore di rocce:")
        print("-" * 50)
        
        if model_exists:
            try:
                model = carica_modello_base()
                print("Modello caricato con successo.")
                print(f"Architettura del modello:")
                model.summary()
                
                if dataset_ready:
                    print(f"\nClassi riconosciute: {subdirs}")
                else:
                    print("\nClassi predefinite: sedimentarie, ignee, metamorfiche")
                
            except Exception as e:
                print(f"Errore nel caricare il modello: {e}")
        else:
            print("Nessun modello addestrato trovato.")
            
        print("\nIstruzioni di utilizzo:")
        print("- Per addestrare: py classificatore_rocce.py train")
        print("- Per classificare: py classificatore_rocce.py predict path/to/image.jpg")
        print("- Per informazioni: py classificatore_rocce.py info")
        
        print("\nVersione di TensorFlow:", tf.__version__)
        print("GPU disponibile:", "Sì" if len(tf.config.list_physical_devices('GPU')) > 0 else "No")
    
    else:
        print(f"\nModalità non valida: {mode}")
        print("Modalità valide: train, predict, info")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperazione interrotta dall'utente.")
    except Exception as e:
        print(f"\nErrore imprevisto: {e}")
    
    print("\nProgramma terminato.")