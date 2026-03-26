import torch
import torch.nn as nn
import torchvision.models as models

def create_encoder(dim_projection=128):
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        self.encoder_q = create_encoder(dim_projection=dim)
        self.encoder_k = create_encoder(dim_projection=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  
            param_k.requires_grad = False     

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

        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            overflow = (ptr + batch_size) - self.K
            self.queue[:, ptr:] = keys.T[:, :batch_size - overflow]
            self.queue[:, :overflow] = keys.T[:, batch_size - overflow:]

        ptr = (ptr + batch_size) % self.K  
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        self._dequeue_and_enqueue(k)

        return logits, labels