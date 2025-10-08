# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:28:00 2025

@author: kehin
"""

import threading
import queue

def get_sync_objects():
    return {
        "data_queue": queue.Queue(),
        "data_lock": threading.Lock(),
        "mdfcm_lock": threading.Lock(),
        "stop_event": threading.Event(),
    }