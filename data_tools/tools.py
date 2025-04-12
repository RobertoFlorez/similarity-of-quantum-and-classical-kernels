import numpy as np
import pandas as pd
import h5py
import glob
import time 
import os


from pathlib import Path
utils_folder = Path(__file__).parent

#make the file importable from the root folder
import sys
sys.path.append(str(utils_folder.parent))



import h5py


def load_feather_folder_as_pd(folder_with_temp_files, short_load = False, initial_key = None):
    """
    Load a folder or a list of feather file paths as a pandas dataframe

    """
    #above line as a loop
    if isinstance(folder_with_temp_files, str):
        import glob
        #if initial key is not None, then we will load only the files that start with initial_key
        if initial_key is not None:
            temp_files = glob.glob(folder_with_temp_files + f"/{initial_key}*.feather")
        else:
            temp_files = glob.glob(folder_with_temp_files + "/*.feather")
    elif isinstance(folder_with_temp_files, list):
        temp_files = folder_with_temp_files
    dicts = []
    times = []
    zero_time = time.time()
    if short_load:
        temp_files = temp_files[:200]
    for idx, temp_file in enumerate(temp_files):    
        try:
            dicts.append(pd.read_feather(temp_file))
        except Exception as e:
            print("Error in file:", temp_file)
            print(e)
        
        times.append(time.time()-zero_time)
        #print(temp_file)
    df = pd.concat(dicts, axis=0, sort=False, ignore_index=True)
    print(time.time()-zero_time)
    return df


def write_dic_results(file_path, dic):
    """Write a dictionary of metadata to an HDF5 group"""
    with h5py.File(file_path, 'a') as file:
        # Create a new group for each experiment
        experiment_group = file.create_group(f"experiment_{len(file)}")
        for key, value in dic.items():
            experiment_group.create_dataset(key, data = value)

def read_experiment_dic_results_simpler(file_path):
    """ 
    This read a dictionary 
    """
    results = []
    with h5py.File(file_path, 'r') as file:
        for group_name in file.keys(): #Group name is experiment_0, experiment_1, etc
            group = file[group_name]
            results.append({}) #Create a new dictionary for each experiment
            for key in group.keys(): #key is K_train, K_test, etc (Matrix names)	
                #print type of group[key] hdf5 object
                results[-1][key] = group[key][()] #Store the matrix and parameters in the dictionary
    return results


def read_experiment_dic_results(file_path, ignore_rho = True, ignore_Ks = True, short_load = False):
    """ 
    This read a dictionary 
    """
    results = []
    with h5py.File(file_path, 'r') as file:
        for group_name in file.keys(): #Group name is experiment_0, experiment_1, etc
            group = file[group_name]
            results.append({}) #Create a new dictionary for each experiment
            if short_load:
                if len(results) > 400:
                    break
            for key in group.keys(): #key is K_train, K_test, etc (Matrix names)	
                if ignore_rho and key in ["density_matrices_train", "density_matrices_test"]:
                    continue
                if ignore_Ks and key in ["K_train", "K_test",  "eigenvectors", ]:
                    continue
                #first let us validate the element
                element = group[key][()] #if element is byte string, then decode it
                if isinstance(element, bytes):
                    pass
                    element = element.decode("utf-8")
                #print type of group[key] hdf5 object
                results[-1][key] = element
    return results

def remove_temporary_files(folder_with_temp_files, format=None):
    """Remove folder with temporary files"""
    import shutil
    if format == "h5":
        for file in glob.glob(folder_with_temp_files + "/*.h5"):
            print("Removing file:", file)
            os.remove(file)
    elif format == None:
        print("Removing folder:", folder_with_temp_files)
        shutil.rmtree(folder_with_temp_files)

def merge_temporary_files(folder_with_temp_files, final_file_path, ignore_errors = False, clean_after_merge = True):
    """Merge temporary HDF5 files into a final file
        folder_with_temp_files: path to folder with temporary files or array of paths to temporary files
        final_file_path: path to final file
    
        Saves the merged file to final_file_path
        """
    #if folder_with_temp_files is a string:
    print("Merging temporary files")
    def split_list(test_list, n_splits):
        # Determine the length of each sublist
        length = len(test_list)
        avg = length / float(n_splits)
        out = []
        last = 0.0

        while last < length:
            out.append(test_list[int(last):int(last + avg)])
            last += avg

        return out

    if isinstance(folder_with_temp_files, str):
        import glob
        temp_files = glob.glob(folder_with_temp_files + "/*.h5")
    elif isinstance(folder_with_temp_files, list):
        temp_files = folder_with_temp_files


    if isinstance(final_file_path, str):
        final_file_path = [final_file_path]
        temp_files_list = [temp_files]
    else:
        temp_files_list = split_list(temp_files, len(final_file_path))
    
    for final_file_path_i, temp_files in zip(final_file_path, temp_files_list):
        with h5py.File(final_file_path_i, 'a') as final_file:
            for temp_file_path in temp_files:
                loaded_experiment_dict = []
                if ignore_errors:
                    try:
                        with h5py.File(temp_file_path, 'r') as file:
                            for group_name in file.keys():
                                group = file[group_name]
                                loaded_experiment_dict.append({})
                                for key in group.keys():
                                    #print type of group[key] hdf5 object
                                    loaded_experiment_dict[-1][key] = group[key][()]
                                write_dic_results(final_file_path_i, loaded_experiment_dict[-1])
                            print("Done with file:", temp_file_path, "Number of experiments:", len(temp_files))
                    except:
                        print("Error in file:", temp_file_path)
                        continue
                else:
                    with h5py.File(temp_file_path, 'r') as file:
                        for group_name in file.keys():
                            group = file[group_name]
                            loaded_experiment_dict.append({})
                            for key in group.keys():
                                #print type of group[key] hdf5 object
                                loaded_experiment_dict[-1][key] = group[key][()]
                        write_dic_results(final_file_path_i, loaded_experiment_dict[-1])
        #Start cleaning the temporary files
    if clean_after_merge:
        for file in temp_files:
            os.remove(file)
        
        

def write_metadata(group, metadata):
    """Write a dictionary of metadata to an HDF5 group"""
    for key, value in metadata.items():
        group.attrs[key] = value

def write_classical_results(file_path, X_train, X_test, y_train, y_test, metadata):
    with h5py.File(file_path, 'a') as file:
        # Create a new group for each experiment
        experiment_group = file.create_group(f"experiment_{len(file)}")

        # Save the 2D arrays
        experiment_group.create_dataset('X_train', data=X_train)
        experiment_group.create_dataset('X_test', data=X_test)
        experiment_group.create_dataset('y_train', data=y_train)
        experiment_group.create_dataset('y_test', data=y_test)

        # Save metadata
        write_metadata(experiment_group, metadata)

def read_metadata(group):
    #create metadata_keys from group.attrs.keys()
    metadata_keys = list(group.attrs.keys())
    return {key: group.attrs.get(key, np.nan) for key in metadata_keys}


def read_experiment_results(file_path):
    results = []
    with h5py.File(file_path, 'r') as file:
        for group_name in file.keys():
            group = file[group_name]
            metadata = read_metadata(group)
            
            results.append({
                'metadata': metadata,
            })


            keys_to_process = {
                "K_train": "K_train",
                "K_test": "K_test",
                "X_train": "X_train",
                "X_test": "X_test",
                "y_train": "y_train",
                "y_test": "y_test",
                "density_matrices_train": "density_matrices_train",
                "density_matrices_test": "density_matrices_test",
            }

            for key, result_key in keys_to_process.items():
                try:
                    results[-1][result_key] = group[key][:]
                except:
                    results[-1][result_key] = np.nan

                



    return results

def rename_kernel(row, quantum = False):
    if quantum:
        if row["encoding_circuit_name"] == "IQPLikeCircuit":
            return "IQP_" + row["method"]
        elif row["encoding_circuit_name"] == "HubregtsenEncodingCircuit":
            return "HZY_CZ_" + row["method"]
        elif row["encoding_circuit_name"] == "YZ_CX_EncodingCircuit":
            return "YZ_CX_" + row["method"]
        elif row["encoding_circuit_name"] == "Separable_rx":
            return "Sep_rx_" + row["method"]
        elif row["encoding_circuit_name"] == "Z_Embedding":
            return "Z_Embedding_" + row["method"]
        else:
            return row["encoding_circuit_name"] + "_"+ row["method"]
    else:
        row_ = row["kernel"]
        poly_string = "poly"
        taylor_string = "taylor_separable_rx" 
        
        if row_ == poly_string or row_ == "rbf_poly":
            row_ += f'_{int(row["degree"])}'
        elif row_ == taylor_string:
            row_ += f'_{int(row["degree"]*2)}'
        else:
            row_ = row["kernel"]    
        return row_

