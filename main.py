from utilities import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video('08fd33_4.mp4')  #read video
    tracker = Tracker('models1/best.pt')
    tracks = tracker.get_objects_tracks(video_frames, read_from_stubs=True,stub_path='stubs/tracks_stub.pkl')  #track objects in video frames

    output_video_frames = tracker.draw_annotations(video_frames, tracks)  #draw annotations on video frames
    save_video(output_video_frames, 'output_video/output_video.avi')  #save video


if __name__== "__main__":
    main()


