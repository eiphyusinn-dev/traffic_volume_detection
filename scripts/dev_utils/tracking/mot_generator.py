import os, cv2

class MOTGenerator:

    def __init__( self, vid_file_path, sequence_number=0 ):
        self.vid_file_path = vid_file_path
        self.seq_num = sequence_number
        self.getFileInfo()

    def getFileInfo( self ):
        
        self.basename = self.vid_file_path.split('/')[-1].split('.')[0]
        self.out_dir  = f"./finetune/tracking/"
        os.makedirs( self.out_dir, exist_ok=True )
        self.gt_file  = f"{self.out_dir}/{self.basename}_{self.seq_num}/gt/"
        os.makedirs( self.gt_file, exist_ok=True )
        self.img_file = f"{self.out_dir}/{self.basename}_{self.seq_num}/img1/"
        os.makedirs( self.img_file, exist_ok=True )
        self.seq_ino  = f"{self.out_dir}/{self.basename}_{self.seq_num}/seqinfo.ini"

    def generateImg( self, cv2_img, frameId ):
        image_filename = f"{frameId:04}.png"
        file_path      = f"{self.img_file}/{image_filename}"
        cv2.imwrite( file_path, cv2_img )

    def generateSeqInfo( self, movie_loader ):
        with open( self.seq_ino, 'w' ) as txt_file:
            txt_file.writelines( f"[Sequence]\n" )
            txt_file.writelines( f"name={movie_loader.movieName}_{self.seq_num}\n" )
            txt_file.writelines( f"imgDir=img1\n" )
            txt_file.writelines( f"frameRate={movie_loader.FPS}\n" )
            txt_file.writelines( f"seqLength={movie_loader.maxMovieFrame}\n" )
            txt_file.writelines( f"imWidth=1920\n" )
            txt_file.writelines( f"imHeight=1080\n" )
            txt_file.writelines( f"imExt=.png\n" )
        txt_file.close()

    



        