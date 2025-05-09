from src.PigMonitor import PigMonitor

monitor = PigMonitor()

if __name__ == "__main__":
    # monitor.monitor("data/farm_videos/D1_S20240812142531_E20240812143353.mp4")
    # monitor.multi_monitor()
    monitor.batch_monitor()
