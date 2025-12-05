import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from web_app import app

class TestAppStartup(unittest.TestCase):
    """
    Verifies that the app's main function runs without errors (e.g. NameError, SyntaxError).
    Mocks the launch method to prevent the server from starting.
    """
    
    @patch('web_app.app.gr.Blocks.launch')
    def test_main_startup(self, mock_launch):
        """Test that app.main() executes and calls launch()."""
        print("\nTesting App Startup (UI Construction)...")
        try:
            app.main()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"app.main() failed to start: {e}")
            
        # Verify launch was called
        self.assertTrue(mock_launch.called, "launch() was not called")
        print("✅ App Startup Passed")

if __name__ == '__main__':
    unittest.main()
