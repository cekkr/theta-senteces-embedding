import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import math
import random

class AngularVector:
    def __init__(self, angles):
        self.angles = angles  # Lista di angoli (tensore)

    def similarity(self, other):
        diff = (self.angles - other.angles) / 2
        return torch.cos(diff) ** 2

    def __repr__(self):
        return f"AngularVector({self.angles.tolist()})"


class HierarchicalAngularEmbedding(nn.Module):
    def __init__(self, pretrained_model_name="sentence-transformers/all-mpnet-base-v2", max_level=4,
                 sentence_limit=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.sentence_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.max_level = max_level
        self.sentence_limit = sentence_limit  # Limite token per frase
        self.angle_transformations = nn.ModuleList()
        for level in range(max_level + 1):
            num_angles = 2**level
            self.angle_transformations.append(nn.Linear(self.sentence_encoder.config.hidden_size, num_angles))
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False #Freeza i parametri, usalo solo per l'embed

    def forward(self, input_text, max_level_override=None):

        sentences = self.split_text(input_text)

        all_level_embeddings = []
        for level in range(max_level_override if max_level_override is not None else self.max_level +1):
            level_embeddings = []

            for sentence in sentences:

              encoded_input = self.tokenizer(sentence, padding='max_length', truncation=True,
                                            return_tensors="pt", max_length = self.sentence_limit)

              with torch.no_grad():
                sentence_embedding = self.sentence_encoder(**encoded_input).last_hidden_state[:, 0, :]

              angles = torch.remainder(torch.sigmoid(self.angle_transformations[level](sentence_embedding)) * math.pi, math.pi)

              level_embeddings.append(AngularVector(angles.squeeze()))

            all_level_embeddings.append(level_embeddings)

        return self.group_embeddings(all_level_embeddings) #Raggruppamento in array ricorsivi

    def split_text(self, text):
        #Divisione "intelligente" del testo
        sentences = []
        current_sentence = []
        tokens = self.tokenizer.tokenize(text)
        for token in tokens:
            current_sentence.append(token)
            if token in ['.', '?', '!', ';'] or len(current_sentence) >= self.sentence_limit - 5:  # -5 per padding
              sentences.append(self.tokenizer.convert_tokens_to_string(current_sentence))
              current_sentence = []
        if current_sentence:  # Aggiungi l'ultima frase, se esiste
            sentences.append(self.tokenizer.convert_tokens_to_string(current_sentence))

        if not sentences:
            sentences = [""] #Se è vuoto, aggiungi stringa vuota

        return sentences


    def group_embeddings(self, all_level_embeddings):
        # Raggruppa gli embedding in una struttura ricorsiva di array
        def recursive_grouping(level, embeddings):
            if level == 0:
                return embeddings
            grouped = []
            for i in range(0, len(embeddings), 2):
                if i + 1 < len(embeddings):
                    grouped.append([embeddings[i], embeddings[i + 1]])
                else:
                    grouped.append([embeddings[i]])  # Se dispari, aggiungi singolarmente
            return recursive_grouping(level - 1, grouped)

        result = []
        for level_embeddings in all_level_embeddings:
          result.append(recursive_grouping(len(level_embeddings)-1, level_embeddings)) #Usa la funzione ricorsiva
        return result

    def compute_similarity(self, embeddings1, embeddings2):
        # Calcola la similarità tra due gerarchie di AngularVector
        def recursive_similarity(group1, group2):
            if isinstance(group1, AngularVector):
                return group1.similarity(group2).mean()

            similarities = []
            for subgroup1, subgroup2 in zip(group1, group2):
                similarities.append(recursive_similarity(subgroup1, subgroup2))

            return sum(similarities) / len(similarities) if similarities else 0.0


        total_similarity = 0.0
        total_weight = 0.0

        for level in range(len(embeddings1)):
          level_similarity = recursive_similarity(embeddings1[level][0], embeddings2[level][0]) #Aggiungi [0]
          weight = 2** level
          total_similarity += level_similarity * weight
          total_weight += weight

        return total_similarity/total_weight if total_weight else 0.0

class CustomLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings1, embeddings2, target):
        similarity = model.compute_similarity(embeddings1, embeddings2)
        loss = torch.mean((1 - target) * torch.pow(similarity, 2) +
                        target * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))
        return loss

# --- Training ---
model = HierarchicalAngularEmbedding()
criterion = CustomLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Dati di esempio: Coppie di frasi e target (1 = simile, -1 = diverso)
dummy_data = [
    ("This is a sentence. This is another sentence.", "These sentences are related. They are quite similar.", 1.0),
    ("Completely unrelated text here.", "Something different altogether.", -1.0),
    ("A long sentence that spans multiple parts, with commas and conjunctions.", "A similar long sentence, broken down into its components.", 1.0),
     ("Short and concise.", "Brief and to the point.", 1.0)
]

num_epochs = 50

for epoch in range(num_epochs):
    total_loss = 0
    for text1, text2, target_val in dummy_data:
        optimizer.zero_grad()
        embeddings1 = model(text1)
        embeddings2 = model(text2)

        target = torch.tensor(target_val, dtype=torch.float32, requires_grad=False)
        loss = criterion(embeddings1, embeddings2, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dummy_data):.4f}")


# --- Inferenza ---
text1 = "This is a test sentence. Let's see how it works."
text2 = "A testing sentence. We want to check the similarity."

with torch.no_grad():
    embeddings1 = model(text1)
    embeddings2 = model(text2, max_level_override=2)  # Forza un livello massimo
    similarity = model.compute_similarity(embeddings1, embeddings2)
    print(f"Similarity: {similarity:.4f}")

print("Embeddings 1:", embeddings1) #Visualizza gli embeddings
print("Embeddings 2:", embeddings2)

# memo: https://aistudio.google.com/prompts/1IRInLrLUF8FqJjfVHE7P4Ue0VU3FaKTE?pli=1