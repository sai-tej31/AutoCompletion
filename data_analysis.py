import matplotlib.pyplot as plt
from main import preprocess_data,create_words

df = pd.read_csv('all-data.csv', encoding='latin-1',header=None)
data = preprocess_data(df)

words = []
for sent in data:
    for word in sent.split(' '):
        if word:
            words.append(word)

from collections import Counter
word_count = Counter(words)

one_gram = sorted(word_count.items(),key = lambda x:x[1],reverse=True)
x_1_gram = [x for (x,y) in one_gram[:10]]
y_1_gram = [y for (x,y) in one_gram[:10]]

fig, ax = plt.subplots(figsize=(10,7))

plt.bar(x_1_gram,y_1_gram,color = 'b')

plt.xlabel("Uni_grams", size=12)
plt.ylabel("Count", size=12)
plt.title("Top 10 Uni grams", size=15)

for index in range(len(x_1_gram)):
    ax.text(x_1_gram[index], y_1_gram[index], y_1_gram[index], size=12,ha = 'center',va = 'baseline')
plt.show()

cleaned_sentences = []
for sent in data:
    words = []
    for word in sent.split(' '):
        if word:
            words.append(word)
    cleaned_sentences.append(' '.join(words))

bi_words = []
for sent in cleaned_sentences:
    senten = sent.split(' ')
    for i in range(len(senten)-1):
        bi_words.append(' '.join(senten[i:i+2]))

from collections import Counter
bi_word_count = Counter(bi_words)



bi_gram = sorted(bi_word_count.items(),key = lambda x:x[1],reverse=True)
x_2_gram = [x for (x,y) in bi_gram[:10]]
y_2_gram = [y for (x,y) in bi_gram[:10]]

fig, ax = plt.subplots(figsize=(10,6))

plt.bar(x_2_gram,y_2_gram,color = 'b')

plt.xlabel("bi_grams", size=12)
plt.ylabel("Count", size=12)
plt.title("Top 10 bi grams", size=15)

for index in range(len(x_2_gram)):
    ax.text(x_2_gram[index], y_2_gram[index], y_2_gram[index], size=12,ha = 'center')
plt.show()

### N-grams without stop words

nltk.download('stopwords')

from collections import Counter
words = [word for word in words if not word.lower() in stopwords.words('english')]
word_count = Counter(words)

one_gram = sorted(word_count.items(),key = lambda x:x[1],reverse=True)
x_1_gram = [x for (x,y) in one_gram[:10]]
y_1_gram = [y for (x,y) in one_gram[:10]]

fig, ax = plt.subplots(figsize=(10,7))

plt.bar(x_1_gram,y_1_gram,color = 'b')

plt.xlabel("one_grams", size=12)
plt.ylabel("Count", size=12)
plt.title("Top 10 one grams", size=15)

for index in range(len(x_1_gram)):
    ax.text(x_1_gram[index], y_1_gram[index], y_1_gram[index], size=12,ha = 'center',va = 'baseline')
plt.show()

if 'The' and 'the' not in stopwords.words('english'):
    print("Yeah")
print('Oops')

cleaned_sentences = []
for sent in data:
    words = []
    for word in sent.split(' '):
        if word:
            if word.lower() not in stopwords.words('english'):
                words.append(word)
    cleaned_sentences.append(' '.join(words))

bi_words = []
for sent in cleaned_sentences:
    senten = sent.split(' ')
    for i in range(len(senten)-1):
        bi_words.append(' '.join(senten[i:i+2]))

bi_word_count = Counter(bi_words)

bi_gram = sorted(bi_word_count.items(),key = lambda x:x[1],reverse=True)
x_2_gram = [x for (x,y) in bi_gram[:10]]
y_2_gram = [y for (x,y) in bi_gram[:10]]

fig, ax = plt.subplots(figsize=(15,6))

plt.bar(x_2_gram,y_2_gram,color = 'b')

plt.xlabel("bi_grams", size=12)
plt.xticks(rotation=45)
plt.ylabel("Count", size=12)
plt.title("Top 10 bi grams", size=15)

for index in range(len(x_2_gram)):
    ax.text(x_2_gram[index], y_2_gram[index], y_2_gram[index], size=12,ha = 'center')
plt.show()

---