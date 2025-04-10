import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(f"Versione di TensorFlow: {tf.__version__}")

def crea_dataset(cartella_base, dimensione_immagine=(150, 150), batch_size=32, split_validation=0.2):
    """
    Crea dataset usando solo funzioni TensorFlow di base
    """
    print(f"Creazione dataset dalla cartella: {cartella_base}")
    
    # Ottieni tutte le sottocartelle (classi)
    classi = [f.name for f in os.scandir(cartella_base) if f.is_dir()]
    print(f"Classi trovate: {classi}")
    
    # Crea un dataset per ciascuna classe
    datasets = []
    for i, classe in enumerate(classi):
        # Percorso della cartella per questa classe
        class_dir = os.path.join(cartella_base, classe)
        
        # Ottieni tutti i file immagine in questa cartella
        files = [os.path.join(class_dir, f.name) for f in os.scandir(class_dir) 
                 if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Mescoliamo casualmente i file
        np.random.shuffle(files)
        
        # Dividi in training e validation
        split_index = int(len(files) * (1 - split_validation))
        training_files = files[:split_index]
        validation_files = files[split_index:]
        
        print(f"Classe {classe}: {len(training_files)} immagini per training, {len(validation_files)} per validation")
        
        # Aggiungi alla lista dei dataset
        datasets.append((training_files, validation_files, i))
    
    # Combina tutte le liste di file
    all_training_files = []
    all_training_labels = []
    all_validation_files = []
    all_validation_labels = []
    
    for train_files, val_files, label_idx in datasets:
        # Crea etichette one-hot
        train_labels = np.zeros((len(train_files), len(classi)))
        train_labels[:, label_idx] = 1
        
        val_labels = np.zeros((len(val_files), len(classi)))
        val_labels[:, label_idx] = 1
        
        all_training_files.extend(train_files)
        all_training_labels.extend(train_labels)
        all_validation_files.extend(val_files)
        all_validation_labels.extend(val_labels)
    
    # Converte le liste in array numpy
    all_training_files = np.array(all_training_files)
    all_training_labels = np.array(all_training_labels)
    all_validation_files = np.array(all_validation_files)
    all_validation_labels = np.array(all_validation_labels)
    
    # Funzione per caricare e preprocessare un'immagine
    def load_and_preprocess_image(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, dimensione_immagine)
        img = tf.cast(img, tf.float32) / 255.0  # Normalizzazione
        return img, label
    
    # Crea i dataset TensorFlow
    train_ds = tf.data.Dataset.from_tensor_slices((all_training_files, all_training_labels))
    train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((all_validation_files, all_validation_labels))
    val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, classi

def crea_modello_semplice(num_classi=3):
    """
    Crea un modello semplice usando l'API sequenziale di TensorFlow
    """
    model = tf.keras.Sequential()
    
    # Primo blocco convoluzionale
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    
    # Secondo blocco convoluzionale
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    
    # Terzo blocco convoluzionale
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    
    # Classifier
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classi, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),  # Uso legacy per compatibilità
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def addestra_modello_base(model, train_ds, val_ds, epoche=20):
    """
    Addestra il modello con impostazioni di base
    """
    print("Inizio addestramento...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epoche,
        verbose=1
    )
    print("Addestramento completato")
    return history

def visualizza_risultati_base(history):
    """
    Visualizza i risultati dell'addestramento
    """
    # Verifichiamo che history contenga i dati necessari
    if not hasattr(history, 'history') or 'accuracy' not in history.history:
        print("Errore: i dati di history non sono disponibili")
        return
    
    plt.figure(figsize=(12, 5))
    
    # Accuratezza
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuratezza del modello')
    plt.ylabel('Accuratezza')
    plt.xlabel('Epoca')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perdita del modello')
    plt.ylabel('Perdita')
    plt.xlabel('Epoca')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

def salva_modello_base(model, path="modello_rocce_base.keras"):
    """
    Salva il modello addestrato
    """
    try:
        model.save(path)
        print(f"Modello salvato in: {path}")
    except Exception as e:
        print(f"Errore nel salvare il modello: {e}")
        # Tenta un formato alternativo
        try:
            model.save(path.replace('.keras', '.h5'))
            print(f"Modello salvato in formato alternativo: {path.replace('.keras', '.h5')}")
        except Exception as e2:
            print(f"Impossibile salvare il modello in nessun formato: {e2}")

def carica_modello_base(path="modello_rocce_base.keras"):
    """
    Carica un modello salvato
    """
    try:
        return tf.keras.models.load_model(path)
    except:
        # Prova un formato alternativo
        try:
            return tf.keras.models.load_model(path.replace('.keras', '.h5'))
        except Exception as e:
            print(f"Impossibile caricare il modello: {e}")
            return None

def predici_immagine_base(model, image_path, class_names, size=(150, 150)):
    """
    Predice la classe di un'immagine
    """
    try:
        # Carica l'immagine
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img_display = tf.image.resize(img, size)
        
        # Prepara per la predizione
        img = tf.cast(img_display, tf.float32) / 255.0
        img = tf.expand_dims(img, 0)  # Aggiungi dimensione batch
        
        # Predizione
        predictions = model.predict(img)
        class_idx = tf.argmax(predictions[0]).numpy()
        
        # Mostra risultato
        class_name = class_names[class_idx]
        confidence = float(predictions[0][class_idx])
        
        print(f"Classe: {class_name}, Confidenza: {confidence:.2f}")
        
        # Visualizza l'immagine
        plt.figure(figsize=(6, 6))
        plt.imshow(img_display.numpy().astype("uint8"))
        plt.title(f"Predizione: {class_name} ({confidence:.2f})")
        plt.axis('off')
        plt.show()
        
        return {
            "class": class_name,
            "confidence": confidence,
            "all_predictions": predictions[0].numpy()
        }
    except Exception as e:
        print(f"Errore nella predizione: {e}")
        return None

def main():
    # Configurazione
    cartella_dataset = "dataset_rocce"
    
    # Verifica l'ambiente
    print("Controllo dell'ambiente TensorFlow...")
    print(f"TensorFlow: {tf.__version__}")
    tf.config.list_physical_devices('GPU')  # Verifica se GPU è disponibile
    
    # Crea i dataset
    try:
        train_ds, val_ds, class_names = crea_dataset(cartella_dataset)
        print(f"Dataset creati con successo. Classi: {class_names}")
        
        # Crea e addestra il modello
        model = crea_modello_semplice(len(class_names))
        history = addestra_modello_base(model, train_ds, val_ds, epoche=15)
        
        # Visualizza risultati
        visualizza_risultati_base(history)
        
        # Salva il modello
        salva_modello_base(model)
        
        print(f"""
        Classificatore di rocce creato con successo!
        
        Per usare il modello su nuove immagini:
          model = carica_modello_base()
          risultato = predici_immagine_base(model, "percorso/immagine.jpg", {class_names})
        """)
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")

if __name__ == "__main__":
    main()