import pandas as pd
import numpy as np
from collections import OrderedDict
from .config_data import ConfigData


def read_header(filename):
    header = {}
    with open(filename, 'r') as f:
        header["metadata"] = f.readline().replace("\n", "").split(",")
        if not header["metadata"][1] == "1.22":
            print("ERROR: version is not 1.22")
        next(f)
        header["marker_type"] = f.readline().replace("\n", "").replace(",,", "frame,time,").split(",")
        header["marker_label"] = f.readline().replace("\n", "").replace(",,", "frame,time,").split(",")
        header["marker_id"] = f.readline().replace("\n", "").replace(",,", "frame,time,").split(",")
        header["header_label1"] = f.readline().replace("\n", "").replace(",,", "frame,time,").split(",")
        header["header_label2"] = f.readline().replace("\n", "").split(",")
    return header


def get_object_names_and_indices(header, quality_metric):
    obj = {}
    names = header["marker_label"]

    if "Quaternion" in header["metadata"]:
        rigid_body_stride = 7 #xyz xyzw
    else:
        rigid_body_stride = 6 #xyz xyz

    marker_stride = 3
    
    if quality_metric == 'Yes':
        rigid_body_stride += 1
        marker_stride += 1
        quality_offset = 1
    else:
        quality_offset = 0

    n_entries = len(names)
    #Get first name
    index = 2 #starting index for first rigidbody
    while index < n_entries:
        name = names[index]  #next elements are known (rigid body always the same number of elements) BUT depends if quaternion or rotation
        #get number of markers
        name_marker = name + ":Marker"
        n_markers = len([s for s in names if name_marker in s]) // marker_stride
        #print(n_markers)
        list_columns = [1]
        list_columns.extend(range(index,index+rigid_body_stride-quality_offset)) #ignore quality

        if "Quaternion" in header["metadata"]:
            list_tuples = [('Time', ''),('RigidBody', 'rx'),('RigidBody', 'ry'),('RigidBody', 'rz'),('RigidBody', 'rw'),
                        ('RigidBody', 'x'),('RigidBody', 'y'),('RigidBody', 'z')]
        else:
            list_tuples = [('Time', ''),('RigidBody', 'rx'),('RigidBody', 'ry'),('RigidBody', 'rz'),
                        ('RigidBody', 'x'),('RigidBody', 'y'),('RigidBody', 'z')]
    
        index += rigid_body_stride
        for i in range(n_markers):
            list_columns.extend(range(index,index+marker_stride-quality_offset))
            index += marker_stride

            marker_name = "Marker" + str(i+1)
            list_tuples.extend([(marker_name,'x'),(marker_name,'y'),(marker_name,'z')])
        
        
        obj[name] = list_columns, list_tuples

    return obj


def read_object_from_file(obj,filename):
    column, name = obj
    column_name = pd.MultiIndex.from_tuples(name)
    
    object_data = pd.read_csv(filename,skiprows=list(range(7)),usecols=column,names=column_name)
    return object_data


def open_data_frame(filename):

    with open(filename, 'r') as f:
        marker_stride = 3
        line1 = f.readline().replace("\n", "").split(",")
        nelements = len(line1)

        if (nelements-2-6) % marker_stride == 0:
            n_markers = (nelements-2-6) // marker_stride
            list_tuples = [('Time', ''),('RigidBody', 'rx'),('RigidBody', 'ry'),('RigidBody', 'rz'),
                        ('RigidBody', 'x'),('RigidBody', 'y'),('RigidBody', 'z')]

        elif (nelements-2-7) % marker_stride == 0:
            n_markers = (nelements-2-7) // marker_stride
            list_tuples = [('Time', ''),('RigidBody', 'rx'),('RigidBody', 'ry'),('RigidBody', 'rz'),('RigidBody', 'rw'),
                        ('RigidBody', 'x'),('RigidBody', 'y'),('RigidBody', 'z')]
        else: print("File Header is wrong!") #TODO break

        for i in range(n_markers):
            marker_name = "Marker" + str(i+1)
            list_tuples.extend([(marker_name,'x'),(marker_name,'y'),(marker_name,'z')])

    column_name = pd.MultiIndex.from_tuples(list_tuples)
    data = pd.read_csv(filename,skiprows=list(range(2)), names=column_name)
    return data



class MotiveLoader:

    def __init__(self, cfg, data, csv):
        self.cfg = cfg
        self.data = data
        self.b_csv = csv

    
    @classmethod
    def from_motive(cls, filename, cfg=None):
      
        obj = OrderedDict() #ensure same order as in file
        header = read_header(filename)

        if cfg is None:
            cfg = ConfigData()
        _obj= get_object_names_and_indices(header,cfg.quality_metric)

        for key, value in _obj.items():
            obj[key] = read_object_from_file(value,filename)

        return cls(cfg,obj,True)


    @classmethod
    def from_data_frame(cls, cfg, filename):
        data = open_data_frame(filename)
        return cls(cfg,data, False)


    def write_data_frames(self, name, filename, start, end):
        self.data[name][start:end].to_csv(filename)
        

    def get_data(self, name=None):
        """
        Returns the data of the object with the given name if data is a motive file
        Otherwise returns the data without a name.
        Complete data is returned
        """

        if self.b_csv:
            return self.data[name]
        else:
            return self.data

    
    def get_data_frame(self, name=None, frame=0):
        """
        Returns a data frame of the object with the given name if data is a motive file
        Otherwise returns the data frame of the data without a name.
        Frame is the ith element of the data sequeunce
        """    

        if self.b_csv:
            return self.data[name].iloc[[frame]]
        else: 
            return self.data.iloc[[frame]]

    
    def get_data_mean(self, name=None):
        """
        Returns the mean of the data of the object with the given name if data is a motive file
        Otherwise returns the mean of the data without a name.
        """

        if self.b_csv:
            return self.data[name].mean(axis=0)
        else:
            return self.data.mean(axis=0)


    def get_motive_names(self):
        keys = []
        for key, value in self.data.items():
            keys.append(key)
        return keys

    
    def get_rigid_body_data_plot(self, name):
        if self.b_csv: #TODO do without this variable
            data = self.data[name]["RigidBody"]
        else:
            data = self.data["RigidBody"]

        index = list(data.index.values)
        return index, data["x"].tolist(), data["y"].tolist(), data["z"].tolist()

    



    #seems to not work this way
    def get_frames_of_resting_object(self, name):
        frame, x, y, z = self.get_object_trajectory(name)

        max_diff = 0.01

        x_diff = np.diff(x)
        y_diff = np.diff(y)
        z_diff = np.diff(z)

        mask_x = np.where(x_diff<max_diff, 1, 0)
        mask_y = np.where(y_diff<max_diff, 1, 0)
        mask_z = np.where(z_diff<max_diff, 1, 0)

        #mask = np.multiply(np.multiply(mask_x,mask_y),mask_z)
        mask = mask_x * mask_y * mask_z
        tmp = []
        indices = []
        for i in range(len(mask_x)):
            #print(i)
            #print(diff[i])

            if mask[i] == 1:
            #if [x for x in (mask_x[i], mask_y[i], mask[z], d) if x is None]:
                tmp.append(i)
            if mask[i] == 0 and len(tmp) > 0:
                t = np.asarray(tmp)
                indices.append(int(np.mean(t)))
                tmp = []
        #for i in range(len(frame)-1):
        #    x[i+1] - x[i]
        return indices
    

    def get_motive_marker(self, name=None, frame=0):
        """
        Returns the marker data of the object with the given name if data is a motive file
        Otherwise returns the marker data of the data without a name.
        Frame is the ith element of the data sequence
        Only returns the marker 1,2,3
        """
        
        data = self.get_data_frame(name, frame)
        c1 = np.array(data["Marker1"])[0]
        c2 = np.array(data["Marker2"])[0]
        c3 = np.array(data["Marker3"])[0]

        return [c1,c2,c3]
    

