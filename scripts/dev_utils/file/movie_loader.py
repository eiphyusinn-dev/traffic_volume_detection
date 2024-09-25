import cv2
from datetime import datetime, timedelta

class MovieLoader:

    @property
    def moviePath( self ):
        return self.__moviePath

    @property
    def cv2Movie( self ):
        return self.__cv2Movie

    @property
    def maxMovieFrame( self ):
        max_frames = int(self.cv2Movie.get(cv2.CAP_PROP_FRAME_COUNT))
        return max_frames

    @property
    def currentFrame( self ):
        return self.__frame

    @property
    def frameId( self ) -> int:
        return self.__frame_id

    @property
    def FPS( self ):
        fps = self.cv2Movie.get(cv2.CAP_PROP_FPS)
        return fps

    def __init__( self, moviePath, start_time, movieName, start_frame=1 ):
        self.__moviePath = moviePath
        self.__cv2Movie = cv2.VideoCapture( self.moviePath )
        self.__frame = None
        self.__frame_id:int = -1
        self.__cv2Movie.set( cv2.CAP_PROP_POS_FRAMES, start_frame )
        self.movieName  = movieName

        self.setupDateTime( start_time )
        self.__checkValidMovie()
        
    def __checkValidMovie( self ):
        if self.maxMovieFrame == 0:
            print(f"Invalid Movie: {self.movieName}")

    def setupDateTime( self, start_time ):
        dateData, timeData        = start_time.split('.')
        year, month, day          = dateData.split(":")
        hour, minute, second      = timeData.split(":")

        self.curr_datetime = datetime(
                int(year), int(month), int(day), int(
                    hour), int(minute), int(second)
            )

    def readFrame( self ):
        res, self.__frame = self.cv2Movie.read()
        self.__frame_id += 1
        # update date time every second
        # print(self.FPS)
        # print(int(self.FPS))
        # print(round(self.FPS))
        if self.frameId % round(self.FPS) == 0:
            self.curr_datetime += timedelta(seconds=1)
