# from shapely import Polygon

class Detection_Results:

    @property
    def bbox( self ) -> list:
        return self.__bbox

    @property
    def confidence( self ) -> float:
        return self.__confidence

    @property
    def class_name( self ) -> str:
        return self.__class_name

    @property
    def x1y1( self ) -> list:
        return list(map(int,tuple(self.bbox[:2])))

    @property
    def x2y2( self ) -> list:
        return list(map(int,tuple(self.bbox[2:4])))

    @property
    def midxY2( self ):
        mid_x = int((self.bbox[2] + self.bbox[0])/2)
        y2    = int( self.bbox[3] )
        return list(map(int, tuple([mid_x, y2 - 5]))) # small value is minus to avoid the bottom of bounding box to be on the line 

    @property
    def class_id( self ) -> int:
        return self.__class_id

    @property
    def tracking_input( self ):
        return [self.x1, self.y1, self.x2, self.y2, self.__confidence, self.__class_id ]
    
    @property
    # def polygonBox( self ):
    #     x1y2 = (self.x1y1[0], self.x2y2[1])
    #     x2y1 = (self.x2y2[0], self.x1y1[1])
    #     polgyon_data = Polygon([(self.x1y1),(x1y2),(self.x2y2),(x2y1)])
    #     return polgyon_data

    def __init__( self, bbox:str, confidence:float, class_name, class_index ):
        self.__bbox: list         = bbox
        self.x1, self.y1, self.x2, self.y2 = self.__bbox
        self.__confidence: float  = float(confidence)
        self.__class_name: str    = class_name
        self.__class_id: int      = class_index
       

    def setIOU( self, iou_value ):
        self.iou = iou_value