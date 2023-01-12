#import os
#import ffmpeg
import argparse
import moviepy.editor as moviepy

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('string', type=str)
    return parser

def convert_avi_to_mp4(avi_file_path, output_name):
    #stream = ffmpeg.input("{input}".format(input = avi_file_path))
    #stream = ffmpeg.hflip(stream)
    #stream = ffmpeg.output(stream, "{output}".format(output = output_name))
    #ffmpeg.run(stream)
    #os.popen("ffmpeg -i {input} {output}".format(input = avi_file_path, output = output_name))
    clip = moviepy.VideoFileClip("{input}".format(input = avi_file_path))
    clip.write_videofile("{output}".format(output = output_name))
    return True

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    input_filename = args.string
    output_filename = input_filename[:-3] + "mp4"
    convert_avi_to_mp4(input_filename,output_filename)



#    stream = ffmpeg.input('input.mp4')
#    stream = ffmpeg.hflip(stream)
#    stream = ffmpeg.output(stream, 'output.mp4')
#    ffmpeg.run(stream)

