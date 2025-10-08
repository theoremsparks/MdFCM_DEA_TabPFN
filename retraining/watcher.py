# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:25:32 2025

@author: kehin
"""

import os

def watch_for_new_data(directory_path, check_interval=60, process_existing_on_start=True, data_queue=None, stop_event=None):
    """
    Watch a directory for new data files. Stops when stop_event is set.
    """
    print(f"\n[Watcher] Watching directory: {directory_path}\n")

    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        known_files = set() if process_existing_on_start else set(os.listdir(directory_path))
    except Exception as e:
        print(f"[Watcher] Init error: {e}")
        known_files = set()

    while not stop_event.is_set():
        try:
            current_files = set(os.listdir(directory_path))
            new_files = current_files - known_files

            for filename in sorted(new_files):
                if filename.lower().endswith('.csv') and 'new_data' in filename.lower():
                    full_path = os.path.join(directory_path, filename)
                    print(f"[Watcher] Found new data file: {filename}")
                    data_queue.put(full_path)

            known_files = current_files
            stop_event.wait(check_interval)

        except Exception as e:
            print(f"[Watcher] Error watching directory: {str(e)}")
            stop_event.wait(check_interval)