# Classificatore di Rocce con Intelligenza Artificiale

## Configurazione rapida (VERSIONE OFFLINE)

### Installazione e configurazione

1. **Scarica i componenti necessari**:
   - Clone o scarica questo repository
   - Scarica il file `pacchetti_offline.zip` dalla [pagina delle release](https://github.com/tuousername/classificatore-rocce/releases)
   - Posiziona `pacchetti_offline.zip` nella directory principale del progetto

2. **Esegui lo script di setup**:
   ```bash
   py setup.py
   ```
   Questo script:
   - Estrarrà automaticamente i pacchetti dall'archivio
   - Creerà un ambiente virtuale
   - Installerà tutte le dipendenze necessarie dai pacchetti locali
   
   **Nota**: Nessuna connessione internet richiesta!

3. **Attiva l'ambiente virtuale**:
   - Windows: `rocce_env\Scripts\activate`
   - Mac/Linux: `source rocce_env/bin/activate`

4. **Esegui il classificatore**:
   ```bash
   py classificatore_rocce.py
   ```

5. **Alternativa semplificata** (solo Windows):
   Fai doppio click sul file `avvia_classificatore.bat`

### Struttura del repository

Il repository è organizzato nel seguente modo:
```
classificatore-rocce/
├── README.md               # Questa guida
├── requirements.txt        # File con le dipendenze
├── classificatore_rocce.py # Il codice principale
├── model.py                # Modulo con le funzioni del modello
├── setup.py                # Script per preparare l'ambiente
├── scarica_pacchetti.py    # Script per gestire i pacchetti offline
├── avvia_classificatore.bat # Script batch per avvio rapido (Windows)
├── pacchetti_offline.zip   # Archivio con tutti i pacchetti necessari (da scaricare)
└── dataset_rocce/          # Cartella per il dataset (inizialmente vuota)
    ├── sedimentarie/       # Immagini di rocce sedimentarie
    ├── ignee/              # Immagini di rocce ignee
    └── metamorfiche/       # Immagini di rocce metamorfiche
```

> **Nota speciale per laboratori**: Il comando per eseguire Python in questo progetto è `py` invece di `python`. Questo è stato configurato appositamente per ambienti di laboratorio Windows dove Python è installato con questo comando.

### Gestione del dataset

Per il dataset di immagini hai due opzioni:

1. **Creare il tuo dataset**:
   - Raccogli immagini di rocce e organizzale nelle tre cartelle all'interno di `dataset_rocce/`
   - Cerca di avere almeno 30-50 immagini per classe

2. **Utilizzare un dataset di esempio**:
   - Se incluso nella distribuzione, troverai immagini di esempio nella cartella `dataset_esempio`
   - Copia le immagini nelle rispettive cartelle all'interno di `dataset_rocce/`

> **Nota per ambienti con proxy**: Questa versione è stata specificamente preparata per funzionare in ambienti con proxy o firewall restrittivi, come laboratori didattici o aziendali. Non è necessaria alcuna connessione internet per installare o utilizzare il software.

---

## Indice
1. [Concetti di base dell'intelligenza artificiale](#concetti-di-base-dellintelligenza-artificiale)
2. [Le reti neurali e come "imparano"](#le-reti-neurali-e-come-imparano)
3. [Le rocce e le loro caratteristiche visive](#le-rocce-e-le-loro-caratteristiche-visive)
4. [Come funziona il nostro modello](#come-funziona-il-nostro-modello)
5. [Guida all'uso del classificatore](#guida-alluso-del-classificatore)
6. [Migliorare il modello](#migliorare-il-modello)
7. [FAQ e problemi comuni](#faq-e-problemi-comuni)
8. [Risorse aggiuntive](#risorse-aggiuntive)

---

## Concetti di base dell'intelligenza artificiale

### Cos'è il machine learning?

Il machine learning (apprendimento automatico) è un ramo dell'intelligenza artificiale che permette ai computer di "imparare" da esempi, senza essere esplicitamente programmati per svolgere un compito specifico. 

**Esempio semplice**: Immagina di dover programmare un computer per riconoscere le mele dalle arance. Con la programmazione tradizionale, dovresti scrivere regole specifiche come "se è rotondo e rosso, è una mela; se è rotondo e arancione, è un'arancia". Ma cosa succede se incontri una mela verde o una rossa con macchie gialle?

Con il machine learning, invece di programmare regole specifiche, "mostri" al computer molti esempi di mele e arance, e lui "impara" da solo a riconoscere i pattern che le distinguono.

### Apprendimento supervisionato

Nel nostro classificatore di rocce utilizziamo l'**apprendimento supervisionato**, che funziona così:

1. Forniamo al modello molti esempi etichettati (immagini di rocce con l'indicazione del tipo)
2. Il modello analizza questi esempi e impara a riconoscere pattern e caratteristiche
3. Quando gli mostriamo una nuova immagine non etichettata, può predire di che tipo di roccia si tratta

È come imparare con un insegnante che ti mostra esempi e ti dice "questa è una roccia sedimentaria", "questa è una roccia ignea", finché non impari a riconoscerle da solo.

---

## Le reti neurali e come "imparano"

### Cos'è una rete neurale artificiale?

Una rete neurale artificiale è un modello computazionale ispirato al funzionamento del cervello umano. È composta da "neuroni" artificiali organizzati in strati che elaborano l'informazione.

**Analogia**: Pensa a una catena di montaggio in una fabbrica. Ogni operaio (neurone) esegue una piccola operazione e passa il risultato all'operaio successivo. Alla fine della catena, otteniamo il prodotto finito (la previsione).

### Reti Neurali Convoluzionali (CNN)

Il nostro classificatore utilizza una **rete neurale convoluzionale**, un tipo di rete specializzata nell'analisi di immagini. Le CNN funzionano in modo simile a come il nostro sistema visivo elabora le informazioni:

1. **Strati convoluzionali**: Cercano caratteristiche semplici come bordi, colori e texture
2. **Strati di pooling**: Riducono la dimensione dell'immagine mantenendo le informazioni importanti
3. **Strati completamente connessi**: Usano le caratteristiche estratte per fare la classificazione finale

**Analogia**: Immagina di osservare una roccia. Prima noti i dettagli più evidenti (colore, forma generale), poi osservi i dettagli più fini (texture, cristalli), e infine, combinando tutte queste osservazioni, determini di che tipo di roccia si tratta.

### Come avviene l'apprendimento

Il processo di apprendimento di una rete neurale avviene attraverso questi passaggi:

1. **Inizializzazione**: La rete inizia con pesi casuali (non sa nulla)
2. **Forward Pass**: Un'immagine viene passata attraverso la rete per ottenere una previsione
3. **Calcolo dell'errore**: Si confronta la previsione con l'etichetta corretta
4. **Backpropagation**: L'errore viene "propagato all'indietro" attraverso la rete
5. **Aggiornamento dei pesi**: I parametri della rete vengono modificati per ridurre l'errore
6. **Ripetizione**: Questo processo viene ripetuto migliaia di volte con molti esempi

Durante l'addestramento, il modello calcola una "funzione di perdita" (loss function) che misura quanto le sue previsioni sono lontane dalla realtà. L'obiettivo è minimizzare questa perdita.

**Analogia**: È come imparare a disegnare copiando un maestro. All'inizio i tuoi disegni saranno molto diversi dall'originale (errore alto), ma con la pratica e correzioni continue migliorerai gradualmente (errore che diminuisce).

---

## Le rocce e le loro caratteristiche visive

### Tipi principali di rocce

Il nostro classificatore è progettato per riconoscere i tre principali tipi di rocce:

1. **Rocce sedimentarie**: Formate dall'accumulo e dalla compattazione di sedimenti, spesso mostrano strati o laminazioni (es. arenaria, calcare, argillite)
2. **Rocce ignee**: Formate dal raffreddamento e dalla solidificazione del magma, spesso presentano cristalli visibili o aspetto vetroso (es. granito, basalto, ossidiana)
3. **Rocce metamorfiche**: Formate dalla trasformazione di altre rocce sotto pressione e temperatura elevate, spesso mostrano bande o foliazione (es. scisto, gneiss, marmo)

### Caratteristiche visive distintive

Il modello di AI impara a riconoscere queste caratteristiche visive:

- **Texture**: Granulare, cristallina, vetrosa, porosa, stratificata
- **Pattern**: Strati, bande, vene, cristalli visibili
- **Colore**: Sebbene il colore da solo non sia determinante, contribuisce alla classificazione
- **Struttura**: Massiva, foliata, vescicolata, frammentaria

**Nota importante**: Proprio come un geologo esperto, il modello non si basa su una singola caratteristica ma sulla combinazione di diverse proprietà visive per fare una classificazione accurata.

---

## Come funziona il nostro modello

### Architettura del modello

Il classificatore utilizza una rete neurale convoluzionale con la seguente struttura:

1. **Strati di input**: Riceve immagini RGB di dimensione 150x150 pixel
2. **Blocchi convoluzionali**: Tre blocchi, ciascuno composto da:
   - Strato convoluzionale (per estrarre caratteristiche)
   - Pooling (per ridurre la dimensionalità)
3. **Strati finali**:
   - Flattening (appiattisce i dati per i layer densi)
   - Dense (512 neuroni con attivazione ReLU)
   - Dropout (previene l'overfitting)
   - Output (3 neuroni con attivazione Softmax per le 3 classi)

### Preprocessamento delle immagini

Prima di entrare nella rete, le immagini vengono preprocessate:
- Ridimensionate a 150x150 pixel
- Normalizzate (valori dei pixel divisi per 255 per avere valori tra 0 e 1)
- Quando necessario, vengono applicate trasformazioni casuali per l'augmentation dei dati

### Data Augmentation

Per migliorare la generalizzazione del modello (cioè la sua capacità di funzionare bene con immagini mai viste), utilizziamo tecniche di "data augmentation" che applicano trasformazioni casuali alle immagini durante l'addestramento:

- Rotazioni
- Zoom
- Flip orizzontali
- Spostamenti
- Cambiamenti di luminosità

Questo permette al modello di imparare a riconoscere le rocce indipendentemente dall'angolazione, dall'illuminazione o dalla distanza.

### Metriche di valutazione

Per valutare la qualità del modello, utilizziamo diverse metriche:

- **Accuratezza**: Percentuale di predizioni corrette
- **Loss (perdita)**: Misura di quanto le predizioni sono lontane dalla realtà
- **Matrice di confusione**: Mostra quali classi vengono confuse tra loro

Durante l'addestramento, monitoriamo queste metriche sia sul set di addestramento che sul set di validazione (immagini che il modello non ha visto durante l'addestramento) per verificare che il modello stia generalizzando bene.

---

## Guida all'uso del classificatore

### Prerequisiti

Per utilizzare il classificatore avrai bisogno di:
- Python 3.8 o superiore
- Tutte le dipendenze sono già incluse nei pacchetti offline

Queste dipendenze saranno installate automaticamente dallo script `setup.py` usando i pacchetti precaricati.

### Struttura del dataset

Per addestrare il modello, avrai bisogno di un dataset organizzato in questo modo:
```
dataset_rocce/
  ├── sedimentarie/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  ├── ignee/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  └── metamorfiche/
      ├── img1.jpg
      ├── img2.jpg
      └── ...
```

### Addestramento del modello

Per addestrare il modello da zero:

```python
# Prepara il dataset
train_ds, val_ds, class_names = crea_dataset("dataset_rocce")

# Crea e addestra il modello
model = crea_modello_semplice(len(class_names))
history = addestra_modello_base(model, train_ds, val_ds, epoche=15)

# Visualizza i risultati
visualizza_risultati_base(history)

# Salva il modello
salva_modello_base(model)
```

### Utilizzo del modello addestrato

Per classificare nuove immagini con un modello già addestrato:

```python
# Carica il modello
model = carica_modello_base()

# Classifica una nuova immagine
risultato = predici_immagine_base(
    model, 
    "percorso/alla/tua/immagine.jpg", 
    ["sedimentarie", "ignee", "metamorfiche"]
)

# Stampa il risultato
print(f"Tipo di roccia: {risultato['class']}")
print(f"Livello di confidenza: {risultato['confidence']:.2f}")
```

---

## Migliorare il modello

### Suggerimenti per un dataset migliore

La qualità del tuo dataset è fondamentale per le prestazioni del modello:

1. **Quantità**: Cerca di avere almeno 100 immagini per classe, più ne hai, meglio è
2. **Diversità**: Includi rocce della stessa classe ma con aspetti diversi (colori, texture, etc.)
3. **Qualità delle immagini**: Usa immagini nitide e ben illuminate
4. **Consistenza**: Scatta le foto da una distanza simile e con illuminazione costante
5. **Sfondo neutro**: Usa uno sfondo neutro o consistente per tutte le immagini
6. **Classificazione corretta**: Assicurati che le immagini siano nelle cartelle corrette

### Esperimenti da provare

Ecco alcuni esperimenti che potresti fare per migliorare il tuo modello:

1. **Modificare l'architettura**: Aggiungere o rimuovere strati convoluzionali
2. **Regolare iperparametri**: Cambiare learning rate, batch size, numero di epoche
3. **Transfer Learning**: Utilizzare un modello pre-addestrato come base (es. VGG16, ResNet)
4. **Regolarizzazione**: Aggiungere più dropout o regolarizzazione L2 per prevenire l'overfitting
5. **Classi aggiuntive**: Espandere il modello per classificare sottotipi di rocce

---

## FAQ e problemi comuni

### Problemi di installazione

**D: Lo script segnala che 'pacchetti_offline.zip' non esiste.**  
R: Devi scaricare il file dalla pagina delle release GitHub e posizionarlo nella directory principale del progetto.

**D: Ricevo errori durante l'estrazione o l'installazione dei pacchetti.**  
R: Assicurati di utilizzare Python 3.8 (consigliato) o versione compatibile. L'archivio potrebbe essere danneggiato, prova a riscaricarlo.

**D: L'ambiente virtuale non si attiva correttamente.**  
R: Su Windows, assicurati di eseguire `rocce_env\Scripts\activate` nel prompt dei comandi. Prova anche a usare il file batch `avvia_classificatore.bat` come alternativa.

### Problemi di addestramento

**D: Il mio modello raggiunge alta accuratezza sul training set ma bassa sul validation set.**  
R: Questo è chiamato "overfitting". Il modello sta "memorizzando" i dati di training invece di imparare pattern generali. Prova ad aggiungere più data augmentation, usa più dropout o raccogli più dati.

**D: L'accuratezza rimane bassa anche dopo molte epoche.**  
R: Verifica che le tue immagini siano correttamente etichettate e che ci sia una chiara distinzione visiva tra le classi. Potresti anche provare un'architettura di rete più complessa.

**D: L'addestramento è troppo lento.**  
R: Prova a ridurre la dimensione delle immagini o il batch size. Se possibile, usa un computer con GPU.

### Problemi di predizione

**D: Il modello classifica quasi tutto come lo stesso tipo di roccia.**  
R: Questo può indicare uno sbilanciamento nel dataset (troppe immagini di un tipo). Cerca di bilanciare il numero di esempi per classe.

**D: Le predizioni sembrano casuali.**  
R: Verifica che il modello sia stato addestrato correttamente e che stia caricando i pesi corretti. Controlla anche la qualità delle tue immagini di test.

### Problemi tecnici

**D: Errori di memoria durante l'addestramento.**  
R: Riduci il batch size o la dimensione delle immagini.

**D: Python non è riconosciuto come comando interno o esterno.**  
R: Assicurati che Python sia installato correttamente e aggiunto al PATH di sistema.

---

## Risorse aggiuntive

### Per approfondire l'intelligenza artificiale

- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) di Google
- [Corso online di Andrew Ng su Coursera](https://www.coursera.org/learn/machine-learning)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) (libro online gratuito)

### Per approfondire la geologia

- [Guida alle rocce e ai minerali](https://geology.com/rocks/)
- [Come si formano le rocce](https://www.nationalgeographic.org/encyclopedia/rock-cycle/)
- [Classificazione delle rocce](https://www.thoughtco.com/types-of-rock-geology-4088094)

### Tutorial e guide di TensorFlow

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/guides/)
- [TensorFlow for Computer Vision](https://www.tensorflow.org/tutorials/images/classification)

---

## Conclusione

Questo classificatore di rocce non è solo uno strumento utile per geologi in erba, ma anche un'ottima introduzione al mondo dell'intelligenza artificiale e del machine learning. Sperimentando con questo progetto, potrai comprendere meglio sia i principi geologici che quelli dell'apprendimento automatico.

Ricorda che l'apprendimento automatico, proprio come lo studio della geologia, richiede pazienza e sperimentazione. Non scoraggiarti se i primi risultati non sono perfetti – ogni esperimento è un'opportunità di apprendimento!

---

*Questo progetto è stato sviluppato per fini educativi e ottimizzato per funzionare in ambienti con accesso internet limitato. Non sostituisce l'expertise di un geologo professionista.*