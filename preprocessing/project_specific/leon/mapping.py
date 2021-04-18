import folium
import numpy as np
from constants import Constants
from shapely.geometry import MultiPolygon, Polygon, mapping
from grid_definition import GridDefinition


class Mapping(object):

    def plotPolygon(self, map_obj, polygon, color):
        folium.GeoJson(polygon, 
            style_function=lambda x: {
                    'color' : color,
                    'weight' : 0.8,
                    'fillColor' : color,
                    'fillOpacity': 0.3
            }
              ).add_to(map_obj)
    
    
    
   # def addToFeatureGroup(self, feature_group, polygon, color):
    #    folium.Ge
        
    
    def plotSensors(self, map_obj, sensorCoords, plotColor):
        for i in range(sensorCoords.shape[0]):
            folium.CircleMarker((sensorCoords[i][0], sensorCoords[i][1]),
                   radius=8,
                    color=plotColor,
                    weight=1.0,
                    fill_color=plotColor,
                    fill=True,
                    fill_opacity=1,
                   ).add_to(map_obj)
    

    def mapExistingSensors(self, map_obj, plotColor):
        constants = Constants()
        existingCoords = constants.getExistingSensorCoords()
        existingNames = constants.getExistingSensorNames()
        #self.plotSensors(map_obj, existingCoords, plotColor)
        for i in range(existingCoords.shape[0]):
            folium.CircleMarker((existingCoords[i][0], existingCoords[i][1]),
                   radius=8,
                    color=plotColor,
                    weight=1.0,
                    fill_color=plotColor,
                    fill=True,
                    fill_opacity=1,
                   ).add_child(folium.Popup(existingNames[i])).add_to(map_obj)

        
    def mapStaticSensors(self, map_obj, plotColor):
        constants = Constants()
        staticCoords = constants.staticSensorCoords()
        plotSensors(map_obj, staticCoords, plotColor)
        for i in range(staticCoords.shape[0]):
            folium.CircleMarker((staticCoords[i][0], staticCoords[i][1]),
                   radius=8,
                    color='#000000',
                    weight=1.0,
                    fill_color='#000000',
                    fill=True,
                    fill_opacity=1,
                   ).add_to(map_obj)
        
    

    def mapPollutionWalk(mapObj, start_date, sid):
        end_date = getEndDate(start_date)
        data_dir = "/Users/zoepetard/Google Drive/Edinburgh/MscProj/FillingTheGaps/data/raw/personal/"+str(start_date)+"-"+str(end_date)+"/"
        sids = ['XXM007', 'XXM008']
        pdata = data_downloader.readAirSpeckPCSV(start_date, end_date, data_dir)
        belt_index = sids.index(sid)
    
        maxPM = np.max(pdata[belt_index]["PM2.5"])
        minPM = np.min(pdata[belt_index]["PM2.5"])
            
    ##Add validation walk
        for j in range(len(pdata[belt_index])):
            folium.CircleMarker((pdata[belt_index]["latitude"][j], pdata[belt_index]["longitude"][j]),
                    radius=5,
                    color='#000000',
                    weight=1.0,
                    fill_color=osm_reader.assignColor(pdata[belt_index]["PM2.5"][j], maxPM, minPM),
                    fill=True,
                    fill_opacity=1.0,
                   ).add_to(mapObj)

    def getEndDate(start_date):
        return start_date + 1
    
    
    def getGeoCells(self):
        grid_definition = GridDefinition()
        grid_definition.init()   
        gridSize = grid_definition.getGridSize()
        topLat = grid_definition.getTopLat()
        bottomLat = grid_definition.getBottomLat()
        leftLon = grid_definition.getLeftLon()
        rightLon = grid_definition.getRightLon()
                    
        height = topLat - bottomLat
        width = rightLon - leftLon

        heightInterval = height / gridSize
        widthInterval = width / gridSize
                    
        cells = []
        for r in range(gridSize):
            top = topLat - r * heightInterval
            bottom = topLat - (r+1) * heightInterval
            for c in range(gridSize):
                cellTuple = []
                left = leftLon + c * widthInterval
                right = leftLon + (c + 1) * widthInterval
                cellTuple.append(tuple([left, top]))
                cellTuple.append(tuple([right, top]))
                cellTuple.append(tuple([right, bottom]))
                cellTuple.append(tuple([left, bottom]))
                cellTuple.append(tuple([left, top]))
                polygon = Polygon(cellTuple)
                m = MultiPolygon([polygon])
                feature = [{'type': 'Feature', 'properties': {'type': 'multipolygon'}, 'geometry': mapping(m)}]
                multi =  {'type': 'FeatureCollection', 'features': feature}
                cells.append(feature)
        return cells