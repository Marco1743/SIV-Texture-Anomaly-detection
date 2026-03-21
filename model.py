import torch
import torch.nn as nn
import torchvision.models as models

def create_encoder(dim_projection=128):
    """
    Crea l'encoder base (ResNet18).
    Modificato per accettare immagini in scala di grigi e per 
    avere un Multi-Layer Perceptron (MLP) finale invece di un classificatore.
    """
    # Carichiamo un ResNet18 non pre-addestrato (partiamo da zero)
    resnet = models.resnet18(weights=None)
    
    # 1. MODIFICA FONDAMENTALE: Il primo layer convoluzionale di default accetta 3 canali (RGB).
    # Le nostre texture sono a 1 canale (Grayscale). Lo sostituiamo:
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 2. PROJECTION HEAD: Sostituiamo il layer di classificazione finale (FC)
    # L'output di ResNet18 è di 512 dimensioni. Lo proiettiamo in uno spazio più piccolo (es. 128)
    dim_mlp = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(dim_mlp, dim_mlp),
        nn.ReLU(),
        nn.Linear(dim_mlp, dim_projection)
    )
    
    return resnet

class MoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.999, T=0.07):
        """
        dim: dimensione delle feature proiettate (es. 128)
        K: grandezza della coda (quante immagini passate ricorda)
        m: coefficiente di momentum per l'aggiornamento dell'encoder fantasma
        T: temperatura per la loss InfoNCE
        """
        super(MoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T

        # Creiamo i due encoder: Quello principale (Query) e quello fantasma (Key)
        self.encoder_q = create_encoder(dim_projection=dim)
        self.encoder_k = create_encoder(dim_projection=dim)

        # Inizializziamo l'encoder Key con gli stessi pesi dell'encoder Query
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # Copia esatta
            param_k.requires_grad = False     # NON si aggiorna con la backpropagation!

        # Creiamo la memoria a coda (Queue)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Aggiorna l'encoder Key lentamente (Media Mobile Esponenziale)
        usando i pesi dell'encoder Query.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Aggiunge le nuove immagini alla coda e butta fuori le più vecchie.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Gestione del puntatore per sovrascrivere i dati più vecchi
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            overflow = (ptr + batch_size) - self.K
            self.queue[:, ptr:] = keys.T[:, :batch_size - overflow]
            self.queue[:, :overflow] = keys.T[:, batch_size - overflow:]

        ptr = (ptr + batch_size) % self.K  # Avvolgimento (Wrap-around)
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: la prima vista dell'immagine (es. equalizzata)
            im_k: la seconda vista dell'immagine (es. sfocata)
        """
        # Calcoliamo le feature della vista 1 (Query)
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        # Calcoliamo le feature della vista 2 (Key) usando l'encoder fantasma
        with torch.no_grad():
            self._momentum_update_key_encoder() # Aggiorniamo lentamente i pesi
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        # Calcoliamo la similarità!
        # Logit positivi: quanto si somigliano im_q e im_k? (La rete deve massimizzarlo)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Logit negativi: quanto si somiglia im_q con tutte le altre migliaia di immagini nella coda? (La rete deve minimizzarlo)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Uniamo i logit in un'unica matrice (Dimensione: Batch Size x (1 + K))
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Applichiamo la temperatura per scalare i valori
        logits /= self.T

        # Le etichette (labels) sono sempre 0, perché il nostro target positivo 
        # (l'immagine gemella) si trova sempre alla colonna 0 della matrice concatenata!
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        # Aggiorniamo la coda con le nuove chiavi estratte
        self._dequeue_and_enqueue(k)

        return logits, labels