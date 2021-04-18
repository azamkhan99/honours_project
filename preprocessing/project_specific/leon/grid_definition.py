from constants import Constants

class GridDefinition(object):
    def init(self):
        constants = Constants()
        self.gridResolution = constants.getGridSize()
    
        self.topLat = constants.corners[0][0]
        self.bottomLat = constants.corners[2][0]
        self.centerLat = self.bottomLat + (self.topLat - self.bottomLat) / 2
        
        self.leftLon = constants.corners[0][1]
        self.rightLon = constants.corners[1][1]
        self.centerLon = self.leftLon + (self.rightLon - self.leftLon) / 2
    
    def getTopLat(self):
        return self.topLat
    
    def getBottomLat(self):
        return self.bottomLat
    
    def getCenterLat(self):
        return self.centerLat
    
    def getLeftLon(self):
        return self.leftLon
    
    def getRightLon(self):
        return self.rightLon
    
    def getCenterLon(self):
        return self.centerLon
    
    def getGridSize(self):
        return self.gridResolution
    
    
    
    
    
                     