from numba import types, typed, typeof


def pyObjToNumbaObj(pyObj):
    if type(pyObj) == dict:
        pyDict = pyObj
        keys = list(pyDict.keys())
        values = list(pyDict.values())
        
        if type(keys[0]) == str:
            nbhKeytype = types.string
        elif type(keys[0]) == int:
            nbhKeytype = types.int64
        elif type(keys[0]) == float:
            nbhKeytype = types.float64
            
        if type(values[0]) == int:
            nbh = typed.Dict.empty(nbhKeytype, types.int64)
            for i,key in enumerate(keys):
                nbh[key] = values[i]
                
            return(nbh)
        
        elif type(values[0]) == str:
            nbh = typed.Dict.empty(nbhKeytype, types.string)
            for i, key in enumerate(keys):
                nbh[key] = values[i]
                
            return(nbh)
        
        elif type(values[0]) == float:
            nbh = typed.Dict.empty(nbhKeytype, types.float64)
            for i, key in enumerate(keys):
                nbh[key] = values[i]
                
            return(nbh)
        
        elif type(values[0]) == dict:
            for i,subDict in enumerate(values):
                subDict=pyObjToNumbaObj(subDict)
                if i == 0:
                    nbh = typed.Dict.empty(nbhKeytype, typeof(subDict))
                    
                nbh[keys[i]] = subDict 
                   
            return(nbh)
        
        elif type(values[0]) == list:
            for i, subList in enumerate(values):
                subList = pyObjToNumbaObj(subList)
                if i == 0:
                    nbh = typed.Dict.empty(nbhKeytype, typeof(subList))
                    
                nbh[keys[i]] = subList
                
            return(nbh)
        
    elif type(pyObj) == list:
        pyList = pyObj
        data = pyList[0]
        if type(data) == int:
            nbs = typed.List.empty_list(types.int64)
            for data_ in pyList:
                nbs.append(data_)
                
            return (nbs)
        
        elif type(data) == str:
            nbs = typed.List.empty_list(types.string)
            for data_ in pyList:
                nbs.append(data_)
                
            return (nbs)
        
        elif type(data) == float:
            nbs = typed.List.empty_list(types.float64)
            for data_ in pyList:
                nbs.append(data_)
                
            return (nbs)
        
        elif type(data) == dict:
            for i,subDict in enumerate(pyList):
                subDict = pyObjToNumbaObj(subDict)
                if i == 0:
                    nbs = typed.List.empty_list(typeof(subDict))
                      
                nbs.append(subDict)
                
            return(nbs)
        
        elif type(data) == list:
            for i,subList in enumerate(pyList):
                subList = pyObjToNumbaObj(subList)
                if i==0:
                    nbs = typed.List.empty_list(typeof(subList))
                     
                nbs.append(subList)
                
            return(nbs)
       

if __name__=="__main__":
    python_dict = {"a": 1, "b":2}
    numba_dict = pyObjToNumbaObj(python_dict)
    print(type(numba_dict))