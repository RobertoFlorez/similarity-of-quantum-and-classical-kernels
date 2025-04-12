import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split  
import pennylane as qml
from data_tools import pennylane_ds
from pmlb import fetch_data

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
utils_folder = Path(__file__).parent

#make the file importable from the root folder
import sys
sys.path.append(str(utils_folder.parent))


#################3 MNIST ############################



###############33 plasticc #####################

def get_plasticc(feature_dimention = None, train_test_split_value = None, num_datapoints  = None, seed = 1):
  """
  Classification dataset 
  as used here:  https://github.com/rsln-s/Importance-of-Kernel-Bandwidth-in-Quantum-Machine-Learning/blob/main/data/plasticc_data/SN_67floats_preprocessed.npy
  train_test_split: float between 0 and 1. If not None, it splits the data into train and test sets. Use 0.25 for example to get 25% of the data for testing
  returns X, y

  We follow the assumption that the dataset is already standarized  and the last column is the label
  if train_test_split is not None, returns X_train, X_test, y_train, y_test
    """
  abs_path = utils_folder / 'input_data' / 'SN_67floats_preprocessed.npy'
  data = np.load(abs_path)
  X = data[:, :-1]
  y = data[:, -1]

  if num_datapoints is not None:
    try:
      np.random.seed(0) #randomly sample num_datapoints from the original dataset
      inds = np.random.choice(range(len(y)), size=num_datapoints, replace=False)
      y = y[inds]
      X = X[inds, :]
    except:
      print("num_datapoints larger than number of datapoints in the dataset")
      return None
  #replace 0 with -1 
  y = y.astype(int)
  y[y == 0] = -1

  #Perform train test split with different seed for cross validation
  n_train = int(train_test_split_value*num_datapoints)
  n_test = int(num_datapoints - n_train)
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train, test_size=n_test, random_state=seed, stratify=y)

  if feature_dimention is not None:
    pca = PCA(n_components=feature_dimention, svd_solver='full', random_state=0)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test
  

def get_dorothea(feature_dimention = None, train_test_split_value = None, num_datapoints  = None, seed = 1):
  """
  Classification dataset 
  as used here:  https://github.com/rsln-s/Importance-of-Kernel-Bandwidth-in-Quantum-Machine-Learning/blob/main/data/plasticc_data/SN_67floats_preprocessed.npy
  train_test_split: float between 0 and 1. If not None, it splits the data into train and test sets. Use 0.25 for example to get 25% of the data for testing
  returns X, y

  We follow the assumption that the dataset is already standarized  and the last column is the label
  if train_test_split is not None, returns X_train, X_test, y_train, y_test
    """
  abs_path = utils_folder / 'input_data' / 'SN_67floats_preprocessed.npy'
  data = np.load(abs_path)

  data = pd.read_csv("./input_data/madelon/MADELON/madelon_train.data", delim_whitespace=True, header=None)
  X = data.to_numpy().astype(float)
  labels = pd.read_csv("./input_data/madelon/MADELON/madelon_train.labels", delim_whitespace=True, header=None, names=["label"])
  labels = labels.to_numpy().astype(int)

  y = labels.flatten()

  if num_datapoints is not None:
    try:
      np.random.seed(0) #randomly sample num_datapoints from the original dataset
      inds = np.random.choice(range(len(y)), size=num_datapoints, replace=False)
      y = y[inds]
      X = X[inds, :]
    except:
      print("num_datapoints larger than number of datapoints in the dataset")
      return None
  #replace 0 with -1 

  #Perform train test split with different seed for cross validation
  n_train = int(train_test_split_value*num_datapoints)
  n_test = int(num_datapoints - n_train)
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train, test_size=n_test, random_state=seed, stratify=y)

  if feature_dimention is not None:
    standar = StandardScaler(with_std=False, with_mean=True)
    X_train = standar.fit_transform(X_train)
    X_test = standar.transform(X_test)

    pca = PCA(n_components=feature_dimention, svd_solver='full', random_state=0)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test
  

def get_higgs(feature_dimention = None, train_test_split_value = None, num_datapoints  = None, seed = 1):
  """
  Classification dataset 
  as used here:  https://github.com/rsln-s/Importance-of-Kernel-Bandwidth-in-Quantum-Machine-Learning/blob/main/data/plasticc_data/SN_67floats_preprocessed.npy
  train_test_split: float between 0 and 1. If not None, it splits the data into train and test sets. Use 0.25 for example to get 25% of the data for testing
  returns X, y

  We follow the assumption that the dataset is already standarized  and the last column is the label
  if train_test_split is not None, returns X_train, X_test, y_train, y_test
    """
  file_path = "./input_data/HIGGS_ds_small.csv"
  # Load the dataset
  data = pd.read_csv(file_path, header = None)
  X = data.iloc[:, 1:].values
  y = data.iloc[:, 0].values

  if num_datapoints is not None:
    try:
      np.random.seed(0) #randomly sample num_datapoints from the original dataset
      inds = np.random.choice(range(len(y)), size=num_datapoints, replace=False)
      y = y[inds]
      X = X[inds, :]
    except:
      print("num_datapoints larger than number of datapoints in the dataset")
      return None
  #replace 0 with -1 
  y = y.astype(int)
  y[y == 0] = -1

  #Perform train test split with different seed for cross validation
  n_train = int(train_test_split_value*num_datapoints)
  n_test = int(num_datapoints - n_train)
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train, test_size=n_test, random_state=seed, stratify=y)

  if feature_dimention is not None:
    pca = PCA(n_components=feature_dimention, svd_solver='full', random_state=0)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test
  



def get_benzene(feature_dimention = None, train_test_split_value = None, num_datapoints  = None, seed = 1):
  """
   Regression dataset as given by Jan, I took the original dataset and obtained 10000 training points using seed 1
    """
  abs_path = utils_folder / 'input_data' / "benzene" / "benzene_N10000_seed1.csv"
  #use pandas to read the csv
  data = pd.read_csv(abs_path)
  data = data.to_numpy()

  X = data[:, :-1]
  y = data[:, -1]

  if num_datapoints is not None:
    try:
      np.random.seed(seed)
      inds = np.random.choice(range(len(y)), size=num_datapoints, replace=False)
      y = y[inds]
      X = X[inds, :]
    except:
      print("num_datapoints larger than number of datapoints in the dataset")
      return None
  #replace 0 with -1
  if feature_dimention is not None:
    pca = PCA(n_components=feature_dimention, svd_solver='full', random_state=0)
    X = StandardScaler(with_std=True).fit_transform(X)
    X = pca.fit_transform(X) 
    #rescale y labels from -1 to 1 by min max
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
  if train_test_split_value is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_test_split_value, random_state=seed)

    return X_train, X_test, y_train, y_test
  return X, y


def get_penn_state_ds(name, feature_dimention = None, train_test_split_value = None, num_datapoints  = None, seed = 1, with_std = True):
  """
   Regression dataset as given by Jan, I took the original dataset and obtained 10000 training points using seed 1
    """
  abs_path = utils_folder / 'input_data' / "penn_state_ds" 
  #use pandas to read the csv
  X, y = fetch_data(name, local_cache_dir=abs_path, return_X_y=True)

  if num_datapoints is not None:
    try:
      np.random.seed(seed)
      inds = np.random.choice(range(len(y)), size=num_datapoints, replace=False)
      y = y[inds]
      X = X[inds, :]
    except ValueError as e:
      print(f"num_datapoints larger than number of datapoints in the dataset, {name} has ", len(y), "datapoints")
      raise e
      

      
  #replace 0 with -1
  if feature_dimention is not None:
    pca = PCA(n_components=feature_dimention, svd_solver='full', random_state=0)
    X = StandardScaler(with_std=with_std).fit_transform(X)
    X = pca.fit_transform(X) 
    #rescale y labels from -1 to 1 by min max
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
  if train_test_split_value is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_test_split_value, random_state=seed)

    return X_train, X_test, y_train, y_test
  return X, y




#####################3 fashion mnist ############################

import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels



class FashionMNIST:
    """Fashion MNIST dataset.

    
    """
    def __init__(
            self, path=os.path.join(os.path.dirname(__file__), 'input_data', 'fashion')):
        self.path = path

    def load_data(self):
        X_train, y_train = load_mnist(self.path, kind='train')
        X_test, y_test = load_mnist(self.path, kind='t10k')

        X_train, X_test = X_train / 255, X_test / 255
        return X_train, y_train, X_test, y_test


    def filter_by_label(self, X_data, y_data, labels):
        if type(labels) is int:
            labels = [labels]
        for n, label in enumerate(labels):
            mask = y_data==label
            X_filtered = X_data[mask]
            y_filtered = y_data[mask]
            if n == 0:
                X_data_out = X_filtered
                y_data_out = y_filtered
            else:
                X_data_out = np.vstack((X_data_out, X_filtered))
                y_data_out = np.hstack((y_data_out, y_filtered))
        return X_data_out, y_data_out
    def get_dataset(self, a, b, feature_dimention=None, train_test_split_value=None, num_datapoints=None, seed = 1, preprocessing = "train_test_with_mean_PCA"):
        """
        Filters by label, 
        Standarize and apply PCA
        """

        X, y, _, _ = self.load_data()

        #Filter by label and remove one hot encoding
        X, y = self.filter_by_label(X, y, [a, b])
        y = y.astype(int)

        y[y == a] = -1
        y[y == b] = 1

        np.random.seed(0) #randomly sample num_datapoints from the original dataset
        inds = np.random.choice(range(len(y)), size=num_datapoints, replace=False)
        y = y[inds]
        X = X[inds, :]
        #Do train test split by selecting num_training_points and num_test_points randomly by indices

        if preprocessing == "train_test_with_mean_PCA":
            if num_datapoints is not None:
                
                num_training_points = np.round((train_test_split_value * num_datapoints), 1).astype(int)
                num_test_points = num_datapoints - num_training_points
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_training_points, test_size=num_test_points, random_state=seed, stratify=y)

            if feature_dimention is not None:
                scaler = StandardScaler(with_std=False, with_mean=True)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                pca = PCA(n_components=feature_dimention, svd_solver='full', random_state=0)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
            return X_train, X_test, y_train, y_test



        # elif preprocessing == "with_mean":
        #     def standarize_and_PCA(X_train, X_test, n_components):
        #         scaler = StandardScaler(with_std=False, with_mean=True)
        #         X_train_standarized = scaler.fit_transform(X_train)
        #         X_test_standarized = scaler.transform(X_test)

        #         pca = PCA(n_components=n_components, svd_solver='full', random_state=0)
        #         X_train_reduced = pca.fit_transform(X_train_standarized)
        #         X_test_reduced = pca.transform(X_test_standarized)

        #         return X_train_reduced, X_test_reduced

        #     X_train_reduced, X_test_reduced = standarize_and_PCA(
        #     X_train, X_test, n_components=feature_dimention)

        #     num_training_points = np.round((train_test_split_value * num_datapoints), 1).astype(int)
        #     num_test_points = num_datapoints - num_training_points

        #     if num_datapoints is not None:
        #         try:
        #             np.random.seed(seed)
        #             inds_y_train = np.random.choice(range(len(y_train)), size=num_training_points, replace=False)
        #             inds_y_test = np.random.choice(range(len(y_test)), size=num_test_points, replace=False)
        #         except:
        #             print("num_datapoints larger than the number of datapoints in the dataset")
        #             return None
        #         X_train = X_train_reduced[inds_y_train]
        #         y_train = y_train[inds_y_train]
        #         X_test = X_test_reduced[inds_y_test]
        #         y_test = y_test[inds_y_test]
        
    
    

class KuzushijiMNIST(FashionMNIST):
    def __init__(self, path=os.path.join(os.path.dirname(__file__), 'input_data', 'hiragana')):
        """
        Initialize the KuzushijiMNIST object.

        Parameters:
        - path (str): The path to the Kuzushiji MNIST data.
        """
        super().__init__(path)


class Benzene(FashionMNIST):
    def __init__(self, path=os.path.join(os.path.dirname(__file__), 'input_data', 'benzene')):
        """
        Initialize the KuzushijiMNIST object.

        Parameters:
        - path (str): The path to the Kuzushiji MNIST data.
        """
        super().__init__(path)
    def load_data(self):
        abs_path = utils_folder / 'input_data' / "benzene" / "benzene_N10000_seed1.csv"
        #use pandas to read the csv
        data = pd.read_csv(abs_path)
        data = data.to_numpy()

        X = data[:, :-1]
        y = data[:, -1]
        return X, y, None, None
    
    def get_dataset(self, feature_dimention=None, train_test_split_value=None, num_datapoints=None, seed = 1, preprocessing = "MinMax_Std"):
        X, y, _, _ = self.load_data()
        #use scikit min max scaler
        #X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        if feature_dimention is not None:
            #do min max scaling to X
            
            X, _ = self.standarize_and_PCA(X, None, n_components=feature_dimention)
            pass
        if train_test_split_value is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_test_split_value, random_state=seed)

            if num_datapoints is not None:
                try:
                    num_training_points = np.round((train_test_split_value * num_datapoints), 1).astype(int)
                    num_test_points = num_datapoints - num_training_points
                    np.random.seed(seed)
                    inds_y_train = np.random.choice(range(len(y_train)), size=num_training_points, replace=False)
                    inds_y_test = np.random.choice(range(len(y_test)), size=num_test_points, replace=False)
                except:
                    print("num_datapoints larger than the number of datapoints in the dataset")
                    return None
            return X_train[inds_y_train], X_test[inds_y_test], y_train[inds_y_train], y_test[inds_y_test]
        return X, y

    


class OriginalMNIST(FashionMNIST):
    def __init__(self, path=os.path.join(os.path.dirname(__file__), 'input_data', 'original_mnist')):
        """
        Initialize the OriginalMNIST object.
        Parameters:
        - path (str): The path to the Kuzushiji MNIST data.
        """
        super().__init__(path)

    



def save_local_MNIST():
  """
    Download MNIST, reshapes as vectors, normalize, and saves as npz
    Old version please use the class
  """
  import tensorflow_datasets as tfds
  num_tot = 50000
  dataset_name = 'MNIST'
  num_classes = 10
  size = 32
  MNIST_ds = tfds.load('mnist', split=tfds.Split.TRAIN).shuffle(1024, seed = 1).batch(50000)

  ds = MNIST_ds

  for i,ex in enumerate(ds):
      Xtf = ex['image']
      image = Xtf.numpy() / 255.0
      X = np.reshape(image, (image.shape[0], image.shape[1]*image.shape[2]*image.shape[3]))
      X = X / np.outer(np.linalg.norm(X, axis = 1), np.ones(X.shape[1]))
      y = ex['label'].numpy()
      y = np.eye(num_classes)[y]
      break
  np.savez_compressed("MNIST.npz", X = X, y = y)


############# Quantum data #####################
        
#create a cache dictionary
quantum_dataset_cache = {}
dataset_cache = {}



#from data_tools.quantum_data import IsingEncoding
def create_quantum_dataset(dataset,
        num_qubits, num_su2_layers=None,
        su2_random_seed=1):
    """ Creates a regression dataset based on dataset_name and the Ising encoding. 
    
    """
    #if dataset_name_or_data_array is a string
    X_train, X_test, y_train, y_test = dataset
    ising_encoding = IsingEncoding(
        num_qubits=num_qubits, num_su2_layers=num_su2_layers, su2_random_seed=su2_random_seed)
    ising_encoding.create_encoding_circuit()

    y_train_encoded = np.array(ising_encoding.calculate_output(X_train))
    y_test_encoded = np.array(ising_encoding.calculate_output(X_test))

    return X_train, X_test, y_train_encoded, y_test_encoded

#############################################

cached_datasets = {}

def transform_y_to_classification(y):
    #50% lowest value for classification and 50% highest value for classification
    y_classification = np.zeros_like(y, dtype=int)
    y_classification[y > np.median(y)] = 1
    y_classification[y < np.median(y)] = 0
    return y_classification


def sin_norm(x, make_classification = False):
    """ 
    For each row of x, returns sin(||x||)
    """
    y = np.sin(3.32*np.linalg.norm(x, axis=1))
    if make_classification:
        y = transform_y_to_classification(y)
    return y
def norm(x, make_classification = False):
    """ for each row of x, returns ||x||"""
    
    y =  np.linalg.norm(x, axis=1)
    if make_classification:
        y = transform_y_to_classification(y)
    return y

def logistic_norm(x, make_classification = False):
    """ for each row of x, returns ||x||"""
    y = 1/(1+np.exp(-np.linalg.norm(x, axis=1)))
    if make_classification:
        y = transform_y_to_classification(y)
    return y

def row_sum(x, make_classification = False):
    """ for each row of x, returns sum(x)"""
    y = np.sum(x, axis=1)
    if make_classification:
        y = transform_y_to_classification(y)
    return y
def row_sum_squared(x, make_classification = False):
    """ for each row of x, returns sum(x^2)"""
    y = np.sum(x**2, axis=1)
    if make_classification:
        y = transform_y_to_classification(y)
    return y

def row_polynomial(x, make_classification = False):
    """ for each item of the row x, return x[0]**0 + x[1]**1 + x[2]**2 + ..."""
    y = np.sum(x**np.arange(x.shape[1]), axis=1)
    if make_classification:
        y = transform_y_to_classification(y)
    return y
    

def simple_cos(x, make_classification = False):
    """ for each row of x, returns cos(x)"""
    return np.cos(3.32*x)



def simple_sin(x, make_classification = False):
    """ for each row of x, returns cos(x)"""
    return np.sin(5.27*x)

def exp_paper(x, make_classification = False):
    """ norm """
    n = x.shape[1]
    return np.exp(-np.linalg.norm(x, axis=1)**2/n**2)


y_generator_dictionary = {"sin_norm": sin_norm,
                            "norm": norm, 
                            "logistic_norm": logistic_norm,
                            "simple_cos": simple_cos, 
                            "simple_sin": simple_sin,
                            "row_sum": row_sum,
                            "row_sum_squared": row_sum_squared,
                            "row_polynomial": row_polynomial, 
                            "exp_paper": exp_paper,
                            }

def gaussian_distribution(x, sigma):
    return np.exp(-0.5*(x/sigma)**2)/sigma * 1/np.sqrt(2*np.pi)

def exponential_distribution(x, scale = 1):
    return np.exp(-x/scale)/scale

def triangular_distribution(x, a = -1, b = 1, c = 0):
    """ f(x) = 2(x-a)/((b-a)*(c-a)) for a <= x <= c
        f(x) = 2*(b-x)/((b-a)*(b-c)) for c <= x <= b
        f(x) = 0 for x < a or x > b    
    """
    density_function = np.zeros_like(x, dtype=float)

    mask1 = np.logical_and(a <= x, x <= c)
    mask2 = np.logical_and(c <= x, x <= b)

    density_function[mask1] = 2 * (x[mask1] - a) / ((b - a) * (c - a))
    density_function[mask2] = 2 * (b - x[mask2]) / ((b - a) * (b - c))

    return density_function

def cosine_distribution(x):
    return (1+np.cos(x))/(2*np.pi)

def uniform_distribution(x):
    # np.ones_like(x)
    return np.ones_like(x)*1/(np.max(x)-np.min(x))

def make_redundant(X_train, X_test, y_train, y_test, num_qubits):
    X_train_redundant = np.zeros((len(X_train), num_qubits))
    X_test_redundant = np.zeros((len(X_test), num_qubits))

    y_train_redundant = np.zeros((len(X_train)))
    y_test_redundant = np.zeros((len(X_test)))

    for i in range(len(X_train)):
        for j in range(num_qubits):
            X_train_redundant[i,j] = X_train[i]

            y_train_redundant[i] = y_train[i]

            if i < len(X_test):
                X_test_redundant[i,j] = X_test[i]
                y_test_redundant[i] = y_test[i]
    X_train, X_test, y_train, y_test = X_train_redundant, X_test_redundant, y_train_redundant, y_test_redundant
    return X_train, X_test, y_train, y_test

distribution_mapping = {"cosine": cosine_distribution, "uniform": uniform_distribution, "gaussian": gaussian_distribution, "exponential": exponential_distribution, "triangular": triangular_distribution}


def sample_from_a_distribution(x_list, weights, num_datapoints, num_components, seed):
    np.random.seed(seed)

    X_sampled = np.random.choice(x_list, num_datapoints*num_components, p=weights/np.sum(weights))
    X = X_sampled
    X_temp = X_sampled[:num_datapoints*num_components]
    X = X_temp.reshape(num_datapoints, num_components)

    return X

def artificial_dataset(num_qubits, num_datapoints, train_test_split_value = 0.80, distribution = "cosine", y_generator_label = "sin_norm", higher_dim_original_distribution = True, optional_pca = True,  use_cache = False, **kwargs):
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split 
    """ 
    Creates an artificial dataset with a given distribution and y_generator_label.

    distribution: str, "cosine", "uniform", "gaussian", "exponential", "triangular"
    y_generator_label: str, "sin_norm", "norm", "simple_cos" or a function that takes an array of shape (num_datapoints, num_qubits) and returns an array of shape (num_datapoints,)

    if higher_dim_original_distribution is True, we sample from a higher fixed dimension = 10 and then do PCA
    if optional_pca is True, we do PCA from the higher dimension to num_qubits
    
    """ 

    dataset_name = f"{distribution}_dataset_{num_qubits}_{num_datapoints}_{train_test_split_value}_{y_generator_label}"

    np.random.seed(1)


    if type(y_generator_label) == str:
        y_generator = y_generator_dictionary[y_generator_label]        
    else:
        y_generator = y_generator_label
    
    if dataset_name in cached_datasets and use_cache:
        X =  cached_datasets[dataset_name]
    else:
        if type(distribution) == str:
            if "redudant" in distribution:
                distribution = distribution.split("_")[1]
                print(distribution)
            if distribution == "cosine":
                x_list = np.linspace(-np.pi, np.pi, 1000) #-2*np.pi, 2*np.pi
                weights = cosine_distribution(x_list)
            if distribution == "extended_cosine":
                x_list = np.linspace(-4*np.pi, 4*np.pi, 1000)
                weights = cosine_distribution(x_list)
            elif distribution == "uniform": 
                x_list = np.linspace(0, 1, 1000) #0, 5
                weights = uniform_distribution(x_list)
            elif distribution == "uniform_pi": 
                x_list = np.linspace(-np.pi, np.pi, 1000) #0, 5
                weights = uniform_distribution(x_list)
            elif distribution == "gaussian":
                x_list = np.linspace(-3, 3, 1000)
                weights = gaussian_distribution(x_list, 1)
            elif distribution == "extended_uniform":
                x_list = np.linspace(-6, 6, 1000)
                weights = uniform_distribution(x_list)
            elif distribution == "exponential":
                x_list = np.linspace(0, 3, 1000)
                weights = exponential_distribution(x_list)
            elif distribution == "triangular":
                x_list = np.linspace(-1, 1, 1000)
                weights = triangular_distribution(x_list)
        else:
            x_list = distribution[0]
            distribution_function = distribution[1]
            weights = distribution_function(x_list)

        if higher_dim_original_distribution:
            #if true, we sample from a higher fixed dimension and then do PCA
            num_components = 6
        else:
            num_components = num_qubits
        X_sampled = np.random.choice(x_list, num_datapoints*num_components, p=weights/np.sum(weights))

        
        X = X_sampled
        X_temp = X_sampled[:num_datapoints*num_components]
        X = X_temp.reshape(num_datapoints, num_components)

            

        #load make_classification from kwargs

        make_classification = False 
        if "make_classification" in kwargs:
            make_classification = kwargs["make_classification"]
        
        y = y_generator(X, make_classification = make_classification)

        if optional_pca:
            #standarize X before 
            #standarizer = StandardScaler()
            #X = standarizer.fit_transform(X)

            pca = PCA(n_components=num_qubits, svd_solver='full', random_state=0)
            X = pca.fit_transform(X)    
        
        if use_cache:
            cached_datasets[dataset_name] = X

    if train_test_split_value is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_test_split_value, random_state=1)

    return X_train, X_test, y_train, y_test

cached_pca_datasets = {}
def load_dataset_without_quantum_cache(dataset_name, num_qubits, num_datapoints, train_test_split_value = 0.80, seed = 1, **kwargs):
    #implement cached for PCA
    if isinstance(dataset_name, bytes):
        dataset_name = dataset_name.decode()
    if dataset_name == "MNIST":
        X_train, X_test, y_train, y_test = OriginalMNIST().get_dataset(a=0, b=1, feature_dimention=num_qubits, train_test_split_value=train_test_split_value, num_datapoints=num_datapoints, seed=seed, **kwargs)
    elif dataset_name == "plasticc":
        X_train, X_test, y_train, y_test = get_plasticc(feature_dimention=num_qubits, train_test_split_value=train_test_split_value, num_datapoints=num_datapoints, seed=seed, **kwargs)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    elif dataset_name == "dorothea":
        X_train, X_test, y_train, y_test = get_dorothea(feature_dimention=num_qubits, train_test_split_value=train_test_split_value, num_datapoints=num_datapoints, seed=seed, **kwargs)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    elif dataset_name == "higgs":
        X_train, X_test, y_train, y_test = get_higgs(feature_dimention=num_qubits, train_test_split_value=train_test_split_value, num_datapoints=num_datapoints, seed=seed)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    elif dataset_name.startswith("fMNIST"):
        a = int(dataset_name.split("fMNIST")[1][0])
        b = int(dataset_name.split("fMNIST")[1][1])
        X_train, X_test, y_train, y_test = FashionMNIST().get_dataset(a=a, b=b, feature_dimention=num_qubits, train_test_split_value=train_test_split_value, num_datapoints=num_datapoints, seed=seed, **kwargs)
    elif dataset_name.startswith("oldkMNIST"):
        a = int(dataset_name.split("kMNIST")[1][0])
        b = int(dataset_name.split("kMNIST")[1][1])
        X_train, X_test, y_train, y_test = KuzushijiMNIST().get_dataset(a=a, b=b, feature_dimention=num_qubits, train_test_split_value=train_test_split_value, num_datapoints=num_datapoints, seed=seed, preprocessing="with_mean", **kwargs)
    elif dataset_name.startswith("kMNIST"):
        a = int(dataset_name.split("kMNIST")[1][0])
        b = int(dataset_name.split("kMNIST")[1][1])
        X_train, X_test, y_train, y_test = KuzushijiMNIST().get_dataset(a=a, b=b, feature_dimention=num_qubits, train_test_split_value=train_test_split_value, num_datapoints=num_datapoints, seed=seed, preprocessing="train_test_with_mean_PCA", **kwargs)
    elif dataset_name.startswith("MNIST"):
        a = int(dataset_name.split("MNIST")[1][0])
        b = int(dataset_name.split("MNIST")[1][1])
        X_train, X_test, y_train, y_test = OriginalMNIST().get_dataset(a=a, b=b, feature_dimention=num_qubits, train_test_split_value=train_test_split_value, num_datapoints=num_datapoints, seed=seed, **kwargs)
    elif dataset_name == "benzene":
        X_train, X_test, y_train, y_test = Benzene().get_dataset(feature_dimention=num_qubits, train_test_split_value=train_test_split_value, num_datapoints=num_datapoints, seed=seed, **kwargs)
    elif "pennylane" in dataset_name: 
        #pennylane_two-curves or pennylane_hidden-manifold
        dataset_name = dataset_name.split("_")[1]
        if dataset_name == "hidden-manifold":
            X, y = pennylane_ds.generate_hidden_manifold_model(num_datapoints, num_qubits, 6, seed = 0)
        elif dataset_name == dataset_name == "two-curves":
            X, y = pennylane_ds.generate_two_curves(num_datapoints, num_qubits, degree = 6,offset=0.2, noise=0.1, seed = 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_test_split_value, random_state=seed)
        standarizer = StandardScaler(with_mean=True, with_std=False)
        X_train = standarizer.fit_transform(X_train)
        X_test = standarizer.transform(X_test)
    elif dataset_name.startswith("PS"):
        dataset_name = dataset_name.split("PS_")[1]
        print(dataset_name)
        X_train, X_test, y_train, y_test = get_penn_state_ds(dataset_name, feature_dimention=num_qubits, num_datapoints=num_datapoints, train_test_split_value=train_test_split_value, seed=seed, **kwargs)
    elif dataset_name == "cosine":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "cosine", **kwargs)
    elif dataset_name == "uniform":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "uniform", **kwargs)
    elif dataset_name == "uniform_pi":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "uniform_pi", **kwargs)
    elif dataset_name == "gaussian":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "gaussian", **kwargs)
    elif dataset_name == "extended_cosine":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "extended_cosine", **kwargs)
    elif dataset_name == "extended_uniform_pi":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "uniform_pi", **kwargs)
    elif dataset_name == "gaussian_simple_cos":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "gaussian", y_generator_label = "simple_cos", **kwargs)
    elif dataset_name == "gaussian_sine_norm":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "gaussian", y_generator_label = "sin_norm", **kwargs)
    elif dataset_name == "exponential":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "exponential", **kwargs)
    elif dataset_name == "triangular":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "triangular", **kwargs)
    elif dataset_name == "triangular_simple_cos":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "triangular", y_generator_label = "simple_cos", **kwargs)
    elif dataset_name == "cosine_sin_norm":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "cosine", y_generator_label = "sin_norm", **kwargs)
    elif dataset_name == "extended_uniform_sin_norm":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "extended_uniform", y_generator_label = "sin_norm", **kwargs)
    elif dataset_name == "extended_uniform_pi_norm":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "uniform_pi", y_generator_label = "sin_norm", **kwargs)
    elif dataset_name == "cosine_norm":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "cosine", y_generator_label = "norm", **kwargs)
    elif dataset_name == "cosine_simple_cos":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "cosine", y_generator_label = "simple_cos", **kwargs)
    elif dataset_name == "uniform_sin_norm":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "uniform", y_generator_label = "sin_norm", **kwargs)
    elif dataset_name == "uniform_norm":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "uniform", y_generator_label = "norm", **kwargs)
    elif dataset_name == "uniform_simple_cos":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "uniform", y_generator_label = "simple_cos", **kwargs)
    elif dataset_name == "uniform_simple_sin":
        X_train, X_test, y_train, y_test = artificial_dataset(num_qubits, num_datapoints, train_test_split_value = train_test_split_value, distribution = "uniform", y_generator_label = "simple_sin", **kwargs)
    elif dataset_name == "redundant_uniform_simple_cos":
        X_train, X_test, y_train, y_test = artificial_dataset(1, num_datapoints, train_test_split_value = train_test_split_value, distribution = "uniform", y_generator_label = "simple_cos", **kwargs)
        X_train, X_test, y_train, y_test = make_redundant(X_train, X_test, y_train, y_test, num_qubits)
    elif dataset_name == "redundant_cosine_simple_cos":
        X_train, X_test, y_train, y_test = artificial_dataset(1, num_datapoints, train_test_split_value = train_test_split_value, distribution = "cosine", y_generator_label = "simple_cos", **kwargs)
        X_train, X_test, y_train, y_test = make_redundant(X_train, X_test, y_train, y_test, num_qubits)
    elif dataset_name == "redundant_gaussian_simple_cos":
        X_train, X_test, y_train, y_test = artificial_dataset(1, num_datapoints, train_test_split_value = train_test_split_value, distribution = "gaussian", y_generator_label = "simple_cos", **kwargs)
        X_train, X_test, y_train, y_test = make_redundant(X_train, X_test, y_train, y_test, num_qubits)
    elif dataset_name == "redundant_extended_cosine_simple_cos":
        X_train, X_test, y_train, y_test = artificial_dataset(1, num_datapoints, train_test_split_value = train_test_split_value, distribution = "extended_cosine", y_generator_label = "simple_cos", **kwargs)
        X_train, X_test, y_train, y_test = make_redundant(X_train, X_test, y_train, y_test, num_qubits)
    


    else:
        if type(dataset_name) == np.ndarray:
            raise ValueError("dataset_name as array not supported yet")
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")
        
    return X_train, X_test, y_train, y_test


def load_dataset(dataset_name, num_qubits, num_datapoints, train_test_split_value = 0.75, seed=1, caching=True, **kwargs):
    """Loads the dataset and returns X_train, X_test, y_train, y_test, 

    dataset_name: str or np.array
    num_qubits: int
    num_datapoints: int
    train_test_split_value: float between 0 and 1. If not None, it splits the data into train and test sets. Use 0.25 for example to get 25% of the data for testing

    valid dataset_names: "MNIST", "MNIST89", "plasticc", "fMNIST03", "kMNIST14"
    or "quantum_MNIST", "quantum_MNIST89", "quantum_plasticc", "quantum_fMNIST03", "quantum_kMNIST14"

    quantum dataset loading takes time, because it runs calculations under the hood.

    If y_generator_label is in the kwargs, it will be used to generate the y values for the original problem with original_problem_num_components=10 components
    """
    print("Loading dataset:", dataset_name)
    #if dataset_name is bytes, convert to string
    if type(dataset_name) == bytes:
        dataset_name = dataset_name.decode()
    if dataset_name.startswith("quantum_"):
        quantum_data = True
        dataset_name_original = dataset_name + f"_{num_qubits}qubits"
        dataset_name = dataset_name.split("_")[1]
    else:
        dataset_name_original = dataset_name + f"_{num_qubits}qubits" + f"_{num_datapoints}points" + f"_{train_test_split_value}train_test" f"_{seed}seed"
        print(dataset_name_original)
        quantum_data = False
    
    if caching ==True:
        if dataset_name_original not in dataset_cache: 
            X_train, X_test, y_train, y_test = load_dataset_without_quantum_cache(dataset_name, num_qubits = num_qubits, num_datapoints = num_datapoints, train_test_split_value = train_test_split_value, seed = seed,**kwargs)    
            print(f"created", dataset_name_original, "seed", seed)
            dataset_cache[dataset_name_original] = X_train, X_test, y_train, y_test
        else:
            print(f"Using", dataset_name_original, "seed", seed)
            return dataset_cache[dataset_name_original]
    else:
        X_train, X_test, y_train, y_test = load_dataset_without_quantum_cache(dataset_name, num_qubits = num_qubits, num_datapoints = num_datapoints, train_test_split_value = train_test_split_value, seed = seed,**kwargs)    

    if quantum_data:
        if dataset_name_original not in quantum_dataset_cache:
            quantum_dataset_cache[dataset_name_original] = create_quantum_dataset((X_train, X_test, y_train, y_test), num_qubits)
            print("Created quantum dataset")
        else:
           print("Using cached quantum dataset")
        return quantum_dataset_cache[dataset_name_original] # X_train, X_test, y_train, y_test

    return X_train, X_test, y_train, y_test

#add names of regression datasets
regression_ds_list = ["benzene", "PS_645_fri_c3_500_50", "PS_294_satellite_image", "PS_4544_GeographicalOriginalofMusic", "PS_574_house_16H", "PS_581_fri_c3_500_25", "PS_582_fri_c1_500_25", "PS_586_fri_c3_1000_25"]
