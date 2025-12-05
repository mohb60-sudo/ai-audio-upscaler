import unittest
import multiprocessing
import queue
import time
import os
from web_app.worker import AudioWorker

class TestAudioWorker(unittest.TestCase):
    def setUp(self):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.worker = AudioWorker(self.task_queue, self.result_queue)
        self.worker.start()
        
    def tearDown(self):
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join()

    def test_worker_startup(self):
        self.assertTrue(self.worker.is_alive())
        
    def test_unknown_task(self):
        # Submit invalid task
        task_id = "test_1"
        self.task_queue.put((task_id, "invalid_task", (), {}))
        
        # Wait for result
        try:
            tid, status, payload = self.result_queue.get(timeout=5)
            self.assertEqual(tid, task_id)
            self.assertEqual(status, "ERROR")
            self.assertIn("Unknown task type", payload)
        except queue.Empty:
            self.fail("Worker did not respond")

    def test_process_batch_empty(self):
        # Submit empty batch
        task_id = "test_batch"
        args = ([], 48000, "Baseline", "Sinc", None, "wav", "output", "cpu", "None", "Balanced", False,
                False, "lr", 0.0, False, True, True, True, 0.0, 1.0, 0.0, False, 8, 0.5, 10, 0.5, False, False, 0.5)
        
        self.task_queue.put((task_id, "process_batch", args, {}))
        
        try:
            tid, status, payload = self.result_queue.get(timeout=10)
            self.assertEqual(tid, task_id)
            self.assertEqual(status, "ERROR")
            self.assertEqual(payload, "No files selected.")
        except queue.Empty:
            self.fail("Worker did not respond")

if __name__ == '__main__':
    unittest.main()
