##Class to read OSM files '.geojson' files and plot polygons or lines using folium
import folium
import json
from shapely.geometry import Point, MultiPolygon, Polygon, mapping
import shapely
import numpy as np
from grid_definition import GridDefinition

class OSMReader(object):
    
    colors = []
    
    def init(self):
        #self.roadColors = ["red", "orange", "yellow", "green", "blue", "purple", "#778899"]
        #self.roadColors = ["orange", "red", "pink", "brown", "purple", "blue", "green", "#808080"]
        #self.roadColors = ["pink", "red", "orange", "yellow", "teal", "green",  "#grey"]
        self.roadColors = ["#E10084", "#FF0D00", "#FF8500", "#FFB700", "#0D8FDC", "#00A543", "#808080"]
        
        self.landColors = ["brown", "purple", "red", "orange", "green", "blue", "#808080"]
        #industrial, commercial, buildings, residential, parks, water, noData
        
    def assignColor(self, pm, maxPM, minPM):
        rangePM = maxPM - minPM
        if pm > 99999:
            return "#000000"
        elif pm < (0.11 * rangePM + minPM):
            color = "#1b7837"
        elif pm < (0.22 * rangePM + minPM):
            color ="#5aae61"
        elif pm < (0.33 * rangePM + minPM):
            color ="#a6dba0"
        elif pm < (0.44 * rangePM + minPM):
            color ="#d9f0d3"
        elif pm < (0.55 * rangePM + minPM):
             color ="#f7f7f7"
        elif pm < (0.66 * rangePM + minPM):
            color = "#e7d4e8"
        elif pm < (0.77 * rangePM + minPM):
            color = "#c2a5cf"
        elif pm < (0.88 * rangePM + minPM):
            color = "#9970ab"
        else:
            color ="#762a83"
        return color
    
    def style_function_land_use(self, feature, num):
        return {
            'fillColor' : self.colors[num],
            'color' : self.colors[num],
            'fillOpacity': 0.3
        }
            

    def addRoadType(self, mapObj, roadGeo, num):
        folium.GeoJson(roadGeo, 
              style_function=lambda x: {
                    'color' : self.roadColors[num],
                    'weight' : 4.0
        }
              ).add_to(mapObj)
    
    def addLandUse2(self, mapObj, landUse, num):
       # colors = ["red", "orange", "yellow", "green", "blue", "purple", "#778899"]
       # fillColor=colors[num]
        #color=colors[num]
        folium.GeoJson(
                landUse, 
                  style_function=self.style_function_land_use(landUse, num)
              ).add_to(mapObj)
        
    def addLandUse(self, mapObj, landUse, num):
        landColors = ["brown", "purple", "red", "orange", "green", "blue", "#808080"]
        
        folium.GeoJson(landUse, 
              style_function=lambda x: {
                    'color' : landColors[num],
                    'weight' : 0.5,
                    'fillColor' : landColors[num],
                    'fillOpacity': 0.3
        }
        ).add_to(mapObj)
        
    def addCellRoadType(self, mapObj, roadGeo, num):
        #roadColors = ["orange", "red", "pink", "brown", "purple", "blue", "green", "#808080"]
        roadColors = ["#E10084", "#FF0D00", "#FF8500", "#FFB700", "#0D8FDC", "#00A543", "#808080"]
        folium.GeoJson(roadGeo, 
              style_function=lambda x: {
                    'color' : roadColors[num],
                    'weight' : 0.5,
                    'fillColor' : roadColors[num],
                    'fillOpacity': 0.3
        }
        ).add_to(mapObj)
        
    def addCellPM(self, mapObj, cell, PM, maxPM, minPM):
        #colors = ["red", "orange", "yellow", "green", "blue", "purple"]
        color = self.assignColor(PM, maxPM, minPM)
        folium.GeoJson(cell, 
              style_function=lambda x: {
                    'color' : color,
                    'fillColor' : color,
                    'fillOpacity': 0.5
        }
              ).add_to(mapObj)
        
    def getRoadGeoClasses(self, lineFilePath):
        # self.roadColors = ["orange", "red", "pink", "brown", "purple", "blue", "green", "#808080"]
        #self.roadColors = ["pink", "red", "orange", "yellow", "blue", "green", "#808080"]
        #self.roadColors = ["#E10084", "#FF0D00", "#FF8500", "#FFB700", "#0D8FDC", "#00A543"]

        with open(lineFilePath) as f:
            lines = json.load(f)
            
        sum = 0
        print(len(lines["features"]))

    
        roads1 = []
        roads2 = []
        roads3 = []
        roads4 = []
        roads5 = []    
        roads6 = []
        #roads7 = []

        for feature in lines["features"]:
            if "highway" in feature["properties"]:
                if (feature["properties"]["highway"] == "motorway"
                   or feature["properties"]["highway"] == "trunk"):
                    roads1.append(feature)
                if (feature["properties"]["highway"] == "primary" ):
                    roads2.append(feature)
                if feature["properties"]["highway"] == "secondary" :
                    roads3.append(feature)
                if (feature["properties"]["highway"] == "tertiary" 
                    or feature["properties"]["highway"] == "tertiary_link" 
                    or feature["properties"]["highway"] == "unclassified" ): 
                    roads4.append(feature)
                #if (feature["properties"]["highway"] == "unclassified" ):
                    #roads5.append(feature)
                if feature["properties"]["highway"] == "residential" or feature["properties"]["highway"] == "service" : 
                    roads5.append(feature) 
                if (feature["properties"]["highway"] == "footway" 
                    or feature["properties"]["highway"] == "cycleway"
                    or feature["properties"]["highway"] == "pedestrian"
                    or feature["properties"]["highway"] == "path"
                    or feature["properties"]["highway"] == "steps"): 
                    roads6.append(feature) 

        roads1geo = {
            'type': 'FeatureCollection',
            'features': roads1
        }

        roads2geo = {'type': 'FeatureCollection', 'features': roads2}
        roads3geo = {'type': 'FeatureCollection', 'features': roads3}
        roads4geo = {'type': 'FeatureCollection', 'features': roads4}
        roads5geo = {'type': 'FeatureCollection', 'features': roads5}
        roads6geo = {'type': 'FeatureCollection', 'features': roads6}
        #roads7geo = {'type': 'FeatureCollection', 'features': roads7}
    
        roadGeos = [roads1geo, roads2geo, roads3geo, roads4geo, roads5geo, roads6geo]#S, roads7geo]
    
        return roadGeos
    
    def getLandGeoClasses(self, multiPolygonFilePath):
        with open(multiPolygonFilePath) as f:
            shapes = json.load(f)
           
        areas1 = [] #industrial
        areas2 = [] #commercial
        areas3 = [] #buildings
        areas4 = [] #residential
        areas5 = [] #parks
        areas6 = [] #water
        #areas7 = [] #car parks (move to middle level ??)

        
        
        sum = 0
        print(len(shapes["features"]))

        for feature in shapes["features"]:
            APPEND = False
            #print("Beginning: " + str(feature))
            #print(feature["properties"])
            #print(len(feature["properties"]))
            if "landuse" in feature["properties"]:
                if feature["properties"]["landuse"] == "industrial" :
                    areas1.append(feature)
                    APPEND = True
                elif (feature["properties"]["landuse"] == "commercial" 
                     or feature["properties"]["landuse"] == "retail" ):
                    areas2.append(feature)
                    APPEND = True
                elif feature["properties"]["landuse"] == "grass" :
                    areas5.append(feature)
                    APPEND = True
                elif feature["properties"]["landuse"] == "residential" :
                    areas4.append(feature)
                    APPEND = True
                #else: APPEND = False
                
            if "leisure" in feature["properties"]:
                if (feature["properties"]["leisure"] == "park" 
                    or feature["properties"]["leisure"] == "garden"
                    or feature["properties"]["leisure"] == "golf_course"
                    or feature["properties"]["leisure"] == "pitch"):
                    areas5.append(feature)
                    APPEND = True
                #else: APPEND = False
                
            if "amenity" in feature["properties"]:
                if feature["properties"]["amenity"] == "grave_yard":
                    areas5.append(feature)
                    APPEND = True
                elif (feature["properties"]["amenity"] == "school"
                    or feature["properties"]["amenity"] == "kindergarten"
                    or feature["properties"]["amenity"] == "college"
                    or feature["properties"]["amenity"] == "university"
                    or feature["properties"]["amenity"] == "hospital"
                    or feature["properties"]["amenity"] == "marketplace"
                    or feature["properties"]["amenity"] == "library"
                    or feature["properties"]["amenity"] == "place_of_worship"):
                    areas2.append(feature)
                    APPEND = True
                #elif (feature["properties"]["amenity"] == "parking"):
                    #areas7.append(feature)
                    #APPEND = True
                #else: APPEND = False
                
            if "building" in feature["properties"]:
                if (feature["properties"]["building"] == "industrial"):
                    areas1.append(feature)
                    APPEND = True
                elif (feature["properties"]["building"] == "commercial" 
                    or feature["properties"]["building"] == "retail"
                    or feature["properties"]["building"] == "office"
                    or feature["properties"]["building"] == "university"
                    or feature["properties"]["building"] == "school"
                    or feature["properties"]["building"] == "kindergarten"
                    or feature["properties"]["building"] == "church"
                    or feature["properties"]["building"] == "museum"
                    or feature["properties"]["building"] == "hospital"
                    or feature["properties"]["building"] == "hotel"):
                    areas2.append(feature)
                    APPEND = True
                elif (feature["properties"]["building"] == "house" 
                    or feature["properties"]["building"] == "apartments"
                    or feature["properties"]["building"] == "residential"
                    or feature["properties"]["building"] == "semidetached_house"
                    or feature["properties"]["building"] == "detached"):
                    areas4.append(feature)
                    APPEND = True
                elif (feature["properties"]["building"] == "yes" and
                    (len(feature["properties"]) == 2 
                        or len(feature["properties"]) == 3 and "type" in feature["properties"])
                    ):
                    areas3.append(feature)
                    APPEND = True
                #else: APPEND = False
                
            if "natural" in feature["properties"]:
                if (feature["properties"]["natural"] == "water"
                    or feature["properties"]["natural"] == "wetland"):
                    areas6.append(feature)
                    APPEND = True
                elif (feature["properties"]["natural"] == "grassland"
                    or feature["properties"]["natural"] == "scrub"
                     or feature["properties"]["natural"] == "wood"):
                    areas5.append(feature)
                    APPEND = True
                    
            if "shop" in feature["properties"]:
                areas2.append(feature)
                APPEND = True
                #print(feature["properties"])
                #else: APPEND = False
            #print(APPEND)    
            if not APPEND: 
                print(feature["properties"])
                sum += 1
                
                
        print("Not appended sum: " + str(sum))    
        areas1geo = {
            'type': 'FeatureCollection',
            'features': areas1
        }

        areas2geo = {'type': 'FeatureCollection', 'features': areas2}
        areas3geo = {'type': 'FeatureCollection', 'features': areas3}
        areas4geo = {'type': 'FeatureCollection', 'features': areas4}
        areas5geo = {'type': 'FeatureCollection', 'features': areas5}
        areas6geo = {'type': 'FeatureCollection', 'features': areas6}
        #areas7geo = {'type': 'FeatureCollection', 'features': areas7}

        areaGeos = [areas1geo, areas2geo, areas3geo, areas4geo, areas5geo, areas6geo]#, areas7geo]
        return areaGeos
                    
    def getCellsWithMajorLU(self):
        areaGeos = self.getLandGeoClasses()
        cells = self.getGeoCells()
                    
        for cell_num, cell in enumerate(cells):
            cell_multi = cell[0]["geometry"]
            cell_shape = shapely.geometry.asShape(cell_multi)
            major_LU = 0
            major_LU_index = 0
        
            for LU_level, areaGeo in enumerate(areaGeos):
                features = areaGeo["features"]
                area = 0
                for feat in range(len(features)):
                    landuse_multi = areaGeo["features"][feat]["geometry"] 
                    landuse_shape = shapely.geometry.asShape(landuse_multi)
                    intersection = cell_shape.intersection(landuse_shape)
                    area += intersection.area
                if (area > major_LU):
                    major_LU = area
                    major_LU_index = LU_level
            if (major_LU_index == 0): #if there are no landuses in the cell, make it the middle land use category
                major_LU_index = 2
            cells[cell_num][0]["properties"]["major_LU"] = major_LU_index
        return cells
               
#To remove from osm reader !! It is in mapping now
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
        
                    