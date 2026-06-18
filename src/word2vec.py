import torch
import torch.nn as nn
import torch.optim as optim

text = "the quick brown fox jumps over the lazy dog"
words = text.split()
unique_words = list(set(words))
wordtoid = {}
idtoword = {}
for i in range(len(unique_words)):
    word = unique_words[i]
    wordtoid[word] = i
    idtoword[i] = word


trainingdata=[]
windowSize=1
for i in range(len(words)):
    targetString=words[i]
    targetID=wordtoid[targetString]
    if i-windowSize>=0:
       leftString=words[i-windowSize]
       leftID=wordtoid[leftString]
       trainingdata.append((targetID,leftID))
    if i+windowSize<len(words):
        rightString=words[i+windowSize]
        rightID=wordtoid[rightString]
        trainingdata.append((targetID,rightID))

class Word2VecModel(nn.Module):
    def __init__(self,vocabSize,embeddingDimension):
        super(Word2VecModel, self).__init__()
        self.embeddings=nn.Embedding(vocabSize,embeddingDimension)
        self.output=nn.Linear(embeddingDimension,vocabSize)
    def forward(self,targetWordID):
        vector=self.embeddings(targetWordID)
        predictionScores=self.output(vector)
        return predictionScores
my_model = Word2VecModel(vocabSize=len(unique_words), embeddingDimension=50) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_model.parameters(), lr=0.1)
epochs = 50

print("\n Start Training")
for epoch in range(epochs):
    totalloss=0
    for targetID,contextID in trainingdata:
        targettensor=torch.tensor([targetID])
        contexttensor=torch.tensor([contextID])
        optimizer.zero_grad()
        predictions = my_model(targettensor)
        loss = criterion(predictions, contexttensor)
        loss.backward()
        optimizer.step()
        totalloss += loss.item()
    if epoch % 10 == 0:
     print("Epoch:", epoch, "| Total Error:", round(totalloss, 4))
 
print("\nTraining Complete!")

print("\n--- Testing the Trained Model ---")


fox_id = torch.tensor([wordtoid["fox"]])
fox_vector = my_model.embeddings(fox_id)

print("The 50-Dimensional Coordinate Vector for 'fox':")
print(fox_vector)


prediction_scores = my_model(fox_id)


best_guess_id = torch.argmax(prediction_scores).item()
best_guess_word = idtoword[best_guess_id]

print("\nIf you say 'fox', the  guessed context word is:", best_guess_word)


    
       
