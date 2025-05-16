from pipeline.video_pipeline import VideoProcessor

if __name__ == "__main__":
    processor = VideoProcessor("config.yaml")
    processor.run()
