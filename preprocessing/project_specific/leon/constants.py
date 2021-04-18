import numpy as np

class Constants(object):
    #The coordinates for the four corners of the grid of interest
    largeCorners = np.array([[21.16137, -101.71261], #NW 21.16137, -101.71261
                    [21.16137, -101.62326], #NE
                    [ 21.07634, -101.62326], #SE 21.07634, -101.62326
                    [21.07634,  -101.71261]]) #SW
    
    mediumCorners = np.array([[21.15, -101.690], #NW 21.16137, -101.71261
                    [21.15, -101.633], #NE
                    [ 21.097, -101.633], #SE 21.07634, -101.62326
                    [21.097,  -101.690]]) #SW
    
    #(smallCorners)
    corners = np.array([[21.135, -101.69], #NW 21.135, -101.69
                    [21.135,-101.661], #NE
                    [ 21.1075, -101.661], #SE 21.1075, -101.661
                    [21.1075, -101.69]]) #SW
    
    gridSize = 60
    
    existingSensorCoords = np.array([[21.101789, -101.634697], #CICEG                                   
                                    [21.109046, -101.688777], #T-21
                                    [21.133894, -101.68025]]) #FM
    existingSensorNames = np.array(["CICEG", 
                                   "T-21 - Hospital General de la Zona 21 ",
                                   "FAM - Facultad de Medicina"])
    
    staticSensorCoords = np.array([[21.10344, -101.63646], #CICEG  
                            [21.10872, -101.68869], #T21
                            [55.9430028,-3.1921472], #200A7CED9D597407 Library 
                            [55.940953, -3.186092], #AA0E63CF5118F98F Tennis court
                            [55.945302, -3.188279], #B61241EF668DBC2C  Bristo Square
                            [55.943014, -3.185994]]) #E786F1568F65C296 Buccleuch Place
    
    #22 days of data collection
    allCollectedDates = [20180629, 20180703, 20180704, 20180705, 20180706, 20180709, 20180710, 20180716, 20180719, 20180723, 20180724, 20180725, 20180726, 20180730, 20180731, 20180801, 20180802, 20180803, 20180806, 20180807, 20180808, 20180809] 
    
    #On 20180725, 20180731 and 20180803, there were issues with the collection
    selectCollectedDates = [20180629, 20180703, 20180704, 20180705, 20180706, 20180709, 20180710, 20180716, 20180719, 20180723, 20180724, 20180726, 20180730, 20180801, 20180802, 20180806, 20180807, 20180808, 20180809] 
    
    #Dates than all ML tuning experiments were run on
    experimentCollectedDates = [20180703, 20180704, 20180705, 20180706, 20180709, 20180710, 20180716, 20180719, 20180723, 20180726]
    
    recentCollectedDates = [20180807, 20180808, 20180809]
    doubleCollectionDates = [20180704, 20180706, 20180802, 20180803] #(but 20180803 is bad)
        
    sixDates = [20180703, 20180706, 20180716, 20180724, 20180726, 20180801]
    thirteenDates = [20180629, 20180703, 20180704, 20180705, 20180706, 20180709, 20180710, 20180716, 20180719, 20180723, 20180726, 20180801, 20180807]
    sixteenDates = [20180629, 20180703, 20180704, 20180705, 20180706, 20180709, 20180710, 20180716, 20180719, 20180723, 20180724, 20180726, 20180801, 20180806, 20180807, 20180808]
    
    def getGridSize(self):
        return self.gridSize
    
    def getCorners(self):
        return self.corners
    
    def getStaticCoords(self):
        return self.staticSensorCoords
    
    def getExistingSensorCoords(self):
        return self.existingSensorCoords
    
    def getExistingSensorNames(self):
        return self.existingSensorNames
    
    def getAllCollectedDates(self):
        return self.allCollectedDates
    
    def getRecentCollectedDates(self):
        return self.recentCollectedDates
    
    def getDoubleCollectedDates(self):
        return self.doubleCollectionDates
    
    def getSelectCollectedDates(self):
        return self.selectCollectedDates
    
    def getExperimentCollectedDates(self):
        return self.experimentCollectedDates
    
    def getSixDates(self):
        return self.sixDates
    
    def getTenDates(self):
        return self.experimentCollectedDates
    
    def getThirteenDates(self):
        return self.thirteenDates
    
    def getSixteenDates(self):
        return self.sixteenDates
    
    def getNineteenDates(self):
        return self.selectCollectedDates
