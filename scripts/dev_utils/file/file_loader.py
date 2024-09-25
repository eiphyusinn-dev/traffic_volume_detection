import os
from file import movie_loader

class FileLoader:

    @property
    def filePath( self ):
        return self.__file_path

    @property
    def fileExtension( self ):
        return self.__extension

    @property
    def validFileTypes( self ):
        return self.__validFileTypes

    @property
    def movieLoader( self ):
        return self.__movieLoader

    def __init__( self, file_path, start_time="2024:03:09.14:00:13", start_frame=1 ):
        self.__file_path      = file_path
        self.__extension      = file_path.suffix
        self.__validFileTypes = [".MP4", ".mp4", ".AVI", ".avi", ".MOV", ".MTS"]
        self.baseName         = file_path.stem
        self.__movieLoader    = movie_loader.MovieLoader( str(self.filePath), start_time, self.baseName, start_frame )

        self.__checkValidFileTypes()

    def __checkValidFileTypes( self ):
        if self.fileExtension not in self.validFileTypes:
            raise ValueError("Invalid file Type")
