import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline
import math
import random

class AngularVector:
    def __init__(self, angles):
        self.angles = angles  # Tensore degli angoli

    def similarity(self, other):
        diff = (self.angles - other.angles) / 2
        return torch.cos(diff) ** 2

    def __repr__(self):
        return f"AngularVector({self.angles.tolist()})"

class ConceptClassifier(nn.Module):
    #Classificatore di concetti per il training
    def __init__(self, pretrained_model_name="bert-base-uncased", num_base_categories=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_base_categories)

        #Congela BERT
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return F.softmax(logits, dim=-1)

class AngularVector:
    def __init__(self, angles):
        self.angles = angles

    def similarity(self, other):
        diff = (self.angles - other.angles) / 2
        return torch.cos(diff) ** 2

    def __repr__(self):
        return f"AngularVector({self.angles.tolist()})"


class AdvancedAngleTransformation(nn.Module):
    #Modulo di trasformazione angolare avanzato
    def __init__(self, input_size, num_angles, num_layers=1, hidden_size=None, activation='relu'):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_size, num_angles))
        else:
            if hidden_size is None:
                hidden_size = (input_size + num_angles) // 2  # Dimensione intermedia di default

            self.layers.append(nn.Linear(input_size, hidden_size))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.Linear(hidden_size, num_angles))

        #Funzione di attivazione
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()  # Nessuna attivazione

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)  # Nessuna attivazione sull'ultimo layer


class HierarchicalAngularEmbedding(nn.Module):
    def __init__(self, pretrained_model_name="sentence-transformers/all-mpnet-base-v2",
                 base_categories=4, max_level=4, sentence_limit=512,
                 angle_transform_layers=1, angle_transform_hidden_size=None,
                 angle_transform_activation='relu', model_save_path="hierarchical_model"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.sentence_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.base_categories = base_categories
        self.max_level = max_level
        self.sentence_limit = sentence_limit
        self.angle_transform_layers = angle_transform_layers
        self.angle_transform_hidden_size = angle_transform_hidden_size
        self.angle_transform_activation = angle_transform_activation
        self.model_save_path = model_save_path

        for param in self.sentence_encoder.parameters():
            param.requires_grad = False

        self.angle_transformations = nn.ModuleList()
        for level in range(max_level + 1):
            num_angles = base_categories * (2 ** level)
            self.angle_transformations.append(
                AdvancedAngleTransformation(self.sentence_encoder.config.hidden_size, num_angles,
                                            num_layers=angle_transform_layers,
                                            hidden_size=angle_transform_hidden_size,
                                            activation=angle_transform_activation)
            )

    def forward(self, input_text, max_level_override=None):
        sentences = self.split_text(input_text)
        all_level_embeddings = []

        for level in range(max_level_override if max_level_override is not None else self.max_level + 1):
            level_embeddings = []
            for sentence in sentences:
                encoded_input = self.tokenizer(sentence, padding='max_length', truncation=True,
                                               return_tensors="pt", max_length=self.sentence_limit)
                with torch.no_grad():
                    sentence_embedding = self.sentence_encoder(**encoded_input).last_hidden_state[:, 0, :]

                angles = torch.remainder(torch.sigmoid(self.angle_transformations[level](sentence_embedding)) * math.pi, math.pi)
                level_embeddings.append(AngularVector(angles.squeeze()))
            all_level_embeddings.append(level_embeddings)

        return self.group_embeddings(all_level_embeddings)

    def split_text(self, text):
      sentences = []
      current_sentence = []
      tokens = self.tokenizer.tokenize(text)
      for token in tokens:
        current_sentence.append(token)
        if token in ['.', '?', '!', ';'] or len(current_sentence) >= self.sentence_limit -5:
          sentences.append(self.tokenizer.convert_tokens_to_string(current_sentence))
          current_sentence = []
      if current_sentence:
        sentences.append(self.tokenizer.convert_tokens_to_string(current_sentence))

      if not sentences:
        sentences = [""]

      return sentences

    def group_embeddings(self, all_level_embeddings):
        def recursive_grouping(level, num_groups, embeddings):
            if level == 0:
                return embeddings
            grouped = []
            group_size = len(embeddings) // num_groups
            for i in range(num_groups):
                start = i * group_size
                end = (i + 1) * group_size if i < num_groups - 1 else len(embeddings)
                grouped.append(embeddings[start:end])
            return recursive_grouping(level - 1, num_groups * 2, grouped)

        result = []
        for level, level_embeddings in enumerate(all_level_embeddings):
            num_groups = self.base_categories * (2 ** level)
            result.append(recursive_grouping(level, self.base_categories, level_embeddings))
        return result

    def compute_similarity(self, embeddings1, embeddings2):
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
            level_similarity = recursive_similarity(embeddings1[level][0], embeddings2[level][0])
            weight = 2 ** level
            total_similarity += level_similarity * weight
            total_weight += weight
        return total_similarity / total_weight if total_weight else 0.0



    def save(self, path=None):
        if path is None:
          path = self.model_save_path

        #Crea la cartella se non esiste
        os.makedirs(path, exist_ok=True)

        #Salva i parametri in un file JSON
        params = {
            'base_categories': self.base_categories,
            'max_level': self.max_level,
            'sentence_limit': self.sentence_limit,
            'angle_transform_layers': self.angle_transform_layers,
            'angle_transform_hidden_size': self.angle_transform_hidden_size,
            'angle_transform_activation': self.angle_transform_activation,
            'pretrained_model_name': self.tokenizer.name_or_path  #Salva il nome, non tutto il tokenizer
        }
        with open(os.path.join(path, 'model_params.json'), 'w') as f:
            json.dump(params, f, indent=4)

        #Salva i pesi dei moduli di trasformazione angolare
        for i, transform in enumerate(self.angle_transformations):
            torch.save(transform.state_dict(), os.path.join(path, f'angle_transform_{i}.pth'))


    @classmethod
    def load(cls, path):
      #Carica i parametri dal file JSON
      with open(os.path.join(path, 'model_params.json'), 'r') as f:
          params = json.load(f)

      #Crea un'istanza della classe con i parametri caricati
      model = cls(**params, model_save_path = path)

      # Carica i pesi
      for i in range(len(model.angle_transformations)):
        state_dict = torch.load(os.path.join(path, f'angle_transform_{i}.pth'))
        model.angle_transformations[i].load_state_dict(state_dict)

      return model

'''
#Salvataggio del modello
model = HierarchicalAngularEmbedding(angle_transform_layers=3, angle_transform_hidden_size=128)
model.save()

#Caricamento del modello
loaded_model = HierarchicalAngularEmbedding.load("hierarchical_model")

#Verifica che i parametri siano stati caricati correttamente
print(loaded_model.angle_transform_layers)
print(loaded_model.angle_transform_hidden_size)
print(loaded_model.angle_transform_activation)
'''

class CustomLoss(nn.Module):
    def __init__(self, margin=0.1, classification_weight=0.5):
        super().__init__()
        self.margin = margin
        self.classification_weight = classification_weight
        self.concept_classifier = ConceptClassifier() #Istanza del classificatore
        self.cross_entropy = nn.CrossEntropyLoss()


    def forward(self, embeddings1, embeddings2, target, text1, text2):
        similarity_loss = self.similarity_loss(embeddings1, embeddings2, target)
        classification_loss = self.classification_loss(text1, text2)

        return (1 - self.classification_weight) * similarity_loss + self.classification_weight * classification_loss


    def similarity_loss(self, embeddings1, embeddings2, target):
        similarity = model.compute_similarity(embeddings1, embeddings2)
        return torch.mean((1 - target) * torch.pow(similarity, 2) +
                            target * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))

    def classification_loss(self, text1, text2):
        #Classificazione e loss per text1
        logits1 = self.concept_classifier(text1)
        target1 = torch.argmax(logits1, dim=-1) #Usa argmax come target

        #Classificazione e loss per text2
        logits2 = self.concept_classifier(text2)
        target2 = torch.argmax(logits2, dim =-1)

        return (self.cross_entropy(logits1, target1) + self.cross_entropy(logits2, target2))/2



# --- Training ---
model = HierarchicalAngularEmbedding()
criterion = CustomLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


dummy_data = [
    ("The cat sat on the mat. It was a fluffy cat.", "A fluffy cat was sitting on the mat.", 1.0),
    ("The sun is shining brightly. Birds are singing.", "It's a sunny day. The birds sing.", 1.0),
    ("The car is red. It has four wheels.", "The bicycle is blue. It has two wheels.", -1.0),
    ("What is the meaning of life?", "The meaning of life is subjective.", 1.0),
     ("He went to the store to buy groceries.", "She purchased food items at the supermarket", 1.0)
]

num_epochs = 50

for epoch in range(num_epochs):
    total_loss = 0
    for text1, text2, target_val in dummy_data:
        optimizer.zero_grad()
        embeddings1 = model(text1)
        embeddings2 = model(text2)

        target = torch.tensor(target_val, dtype=torch.float32, requires_grad=False)
        loss = criterion(embeddings1, embeddings2, target, text1, text2)  # Passa anche i testi
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dummy_data):.4f}")

# --- Inferenza ---
text1 = "The dog barked at the mailman. He was very angry."
text2 = "A dog was barking. The mailman was the target."

with torch.no_grad():
    embeddings1 = model(text1)
    embeddings2 = model(text2, max_level_override=3)
    similarity = model.compute_similarity(embeddings1, embeddings2)
    print(f"Similarity: {similarity:.4f}")
print("Embeddings1", embeddings1)
print("Embeddings2", embeddings2)