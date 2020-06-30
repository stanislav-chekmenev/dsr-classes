import collections
from datetime import datetime
from datetime import timedelta

# Intro
# Ex 1
print('*' * 5)
for _ in range(6):
    print('  *  ')
    
# Ex 2
for i in range(1, 10):
    print(str(i) * i)
   

# Ex 3
l = []
trange = datetime(2019, 9, 25, 7, 5) + timedelta(0, 300) - datetime(2019, 9, 23, 17, 45)
for i in range(0, trange.days * 24 * 3600 + trange.seconds, 300):
    l.append(datetime(2019, 9, 23, 17, 45) + timedelta(0, i))
    
# Ex 4
start = datetime(1989, 9, 9)
while start <= datetime(1990, 10, 3):
    start += timedelta(5)
    
print(start - datetime(1990, 10, 3))



# Ex 5
start = datetime(1988, 2, 28)
trange = datetime(1989, 9, 9) + timedelta(25) - datetime(1988, 2, 28)
mon = start.month

for i in range(0, trange.days, 25):
    tmp = start + timedelta(i)
    if tmp.month != mon:
        print(tmp.month)
    else:
        print('No change')
    mon = tmp.month
        
print(tmp - datetime(1989, 9, 9))



# pep8
def get_data(fold, seq_length): ## cnn only part
    name = '/home/adam/Documents/fold' + str(fold) + '_X.npy'
    x = np.load(name)
    name = '/home/adam/Documents/fold' + str(fold) + '_Y.npy'
    y = np.load(name)
    
    b = a + x
    
    xx = np.array(np.split(x, x.shape[0] // 3, axis=0))
    shuffle = np.random.permutation(len(xx))  
    xx = xx[shuffle]
    y = y[shuffle]
    x = []
    for i in range(len(xx)):
        tmp = xx[i]
        for j in np.arange(0, 30000, 3000):
            x.append(tmp[:, j:j + 3000])  
    
    yy = np.split(y, y.shape[0], axis=0)
    y = []
    for i in range(len(yy)):
        tmp = yy[i]
        for j in np.arange(0, 30000, 3000):
            y.append(tmp[0][j])  

    y = np.array(y)
    y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
    x = np.split(np.array(x), len(x) // seq_length, axis=0)
    y = np.split(np.array(y), len(y) // seq_length, axis=0)
    # per_cat=np.sum(y,axis=0)
    return np.array(x), np.array(y) 


class model:
    def __init__ (self):
        self.m1 = cnn_model1()
        self.m2 = cnn_model2()

        
def cnn_model1(name, rate, x_shaped):
    input = Input(shape=x_shaped)
    sum = 1 + 2
    my_cnn = Conv2D(filters=64, kernel_size=(50, 1), strides=(8, 1), padding="same", input_shape=x_shaped, activation="relu")(input)
    my_cnn = MaxPool2D(pool_size=(8, 1), strides=(8, 1), padding="same")(my_cnn)
    my_cnn = Dropout(rate=rate)(my_cnn)
    for _ in range(3):
        my_cnn = Conv2D(filters=40, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(my_cnn)
        my_cnn = MaxPool2D (pool_size=(4, 1), strides=(4, 1), padding="same")(my_cnn)
        my_cnn = Flatten()(my_cnn)
        my_cnn = Model(inputs=input, outputs=my_cnn)

    return my_cnn

def cnn_model2(name, rate, x_shaped):
    input = Input(shape=x_shaped)
    my_cnn = TimeDistributed(input)
    my_cnn = Conv2D(filters=64, kernel_size=(400, 1), strides=(50, 1), padding="same", input_shape=x_shaped, activation="relu")(input)
    my_cnn = MaxPool2D(pool_size=(4, 1), strides=(4, 1), padding="same")(my_cnn)
    my_cnn = Dropout(rate=rate)(my_cnn)
    for _ in range(3):
        my_cnn = Conv2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(my_cnn)
        my_cnn = MaxPool2D (pool_size=(2, 1), strides=(2, 1), padding="same")(my_cnn)
        my_cnn = Flatten()(my_cnn)
        my_cnn = Model(inputs=input,outputs = my_cnn)
    return my_cnn










# Strings
# Ex 1
def check(char, word):
    return char.lower() in word.lower()

# Ex 2
def stem(sample):
    words = sample.replace(',', '').replace('.','').replace('(','').replace(')','').lower().split()
    stems = []
    
    for word in words:
        if len(word) != 1:
            if len(word) < 4:
                stems.append(word)
            else:
                stems.append(word[:4])
                
    return stems

stems = stem(sample)

stems_dict = collections.defaultdict(int)

for stem in stems:
    stems_dict[stem] += 1
    
# Paths-and-importing
# Version with os
home = os.environ['HOME']
path = os.path.join(home, 'practice')

os.makedirs(path, exist_ok=True)

for i in range(10):
    # Folders
    path_tmp = os.path.join(path, f'{i}')
    os.makedirs(path_tmp, exist_ok=True)
    # Files
    path_file_tmp = os.path.join(path_tmp, f'{i * 2}.py')
    open(path_file_tmp, 'a').close()
    

for root, dirs, files in os.walk(path):
    for file in files:
        if int(file.replace('.py', '')) % 4 == 0:
            old_path = os.path.join(root, file)
            new_path = os.path.join(path, file)
            os.replace(old_path, new_path)
        else:
            os.remove(os.path.join(root, file))

for root, dirs, _ in os.walk(path):
    for d in dirs:
        os.rmdir(os.path.join(root, d))
        
# Version wiht Path
home = os.environ['HOME']
path = os.path.join(home, 'practice')

p = Path(path)
p.mkdir(exist_ok=True)

for i in range(10):
    p.joinpath(f'{i}').mkdir(exist_ok=True)
    p.joinpath(f'{i}',f'{i * 2}.py').touch(exist_ok=True)
        

        
        
        
        
        
# Iterables and files
# Ex 1
for col in colors:
    for size in sizes:
        print(col, size)
        
        
        
        
        
    
        

# Dicts-and-sets
# Ex 1
for k, v in zip(fitz.values(), fitz.keys()):
    print(k, v)
    
# Ex 2
with open('./quotes.json', 'w') as fi:
    json.dump(quotes, fi)
    
# Ex 3
authors = []
words = []


for element in quotes:
    for k, v in element.items():
        if k == 'author':
            authors.append(v.lower())
        else:
            for word in v.lower().split():
                words.append(word)
            
print(f'Unique authors are {set(authors)}')
print(f'Unique words are {set(words)}')

fitz = []
emerson = []

for element in quotes:
    if 'F. SCOTT FITZGERALD'.lower() in element['author'].lower():
        for word in element['text'].lower().split():
            fitz.append(word)
    elif 'RALPH WALDO EMERSON'.lower() in element['author'].lower():
        for word in element['text'].lower().split():
            emerson.append(word)
        
print(set(fitz).intersection(set(emerson)))

# Functions
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
    

# Classes
# Ex 1
class Actor():
    
    def __init__(self, name, num_films=0):
        self.name = name
        self.num_films = num_films
    
    def act(self, films):
        
        for fi in films:
            if self.name in fi.actors:
                self.num_films += 1
                
        return self.num_films
    
    
# Ex 2
# Version 1
class Dataset:
    """Insert docstring here"""
    
    def __init__(self, samples):
        self.samples = samples
    
    def sample_batch(self, batch_size):
        batch = random.sample(self.samples, k=batch_size)
        return batch
        
    

class OrderedDataset(Dataset):
    """Insert docstring here"""
    
    def __init__(self, samples):
        super().__init__(samples)
        
    
    def order_sampled_batch(self, batch_size):
        batch = super().sample_batch(batch_size)
        batch.sort()
        return batch
    
    
    
class ShuffledDataset(Dataset):
    """Insert docstring here"""
        
    def __init__(self, samples):
        super().__init__(samples)
        
    
    def shuffle_sampled_batch(self, batch_size):
        batch = super().sample_batch(batch_size)
        random.shuffle(batch)
        return batch 

# Version 2
class Dataset:
    “”"Insert docstring here”“”
    def __init__(self, samples):
        self.samples = samples
        
    def sample_batch(self, batch_size=10):
        return random.sample(self.samples, batch_size)
    
class OrderedDataset(Dataset):
    “”"Insert docstring here”“”
    def __init__(self, samples):
        samples.sort()
        super().__init__(samples)
        
class ShuffledDataset(Dataset):
    “”"Insert docstring here”“”
    def __init__(self, samples):
        random.shuffle(samples)
        super().__init__(samples)