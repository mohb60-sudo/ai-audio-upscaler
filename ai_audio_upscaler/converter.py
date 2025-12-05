import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)

class AudioConverter:
    """
    Handles batch audio format conversion using FFmpeg.
    Supports recursive scanning, folder structure preservation, and metadata copying.
    """
    
    SUPPORTED_FORMATS = {
        "FLAC": ".flac",
        "WAV": ".wav",
        "MP3": ".mp3",
        "OGG": ".ogg",
        "M4A": ".m4a",
        "ALAC": ".m4a"
    }
    
    @staticmethod
    def check_ffmpeg():
        """Checks if FFmpeg is available in the system PATH."""
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def convert_batch(self, 
                      source_dir: str, 
                      output_dir: Optional[str], 
                      input_filter: str, 
                      output_format: str, 
                      progress_callback: Optional[Callable] = None) -> List[str]:
        """
        Recursively converts files from source_dir.
        
        Args:
            source_dir: Root directory to scan.
            output_dir: Destination root. If None, saves to source_dir.
            input_filter: File extension to filter (e.g., ".m4a") or "All".
            output_format: Target format key (e.g., "FLAC").
            progress_callback: Function(progress_float, message_str).
            
        Returns:
            List of log messages.
        """
        logs = []
        
        if not self.check_ffmpeg():
            return ["❌ Error: FFmpeg not found. Please install FFmpeg and add it to PATH."]
            
        source_path = Path(source_dir)
        if not source_path.exists():
            return ["❌ Error: Source directory does not exist."]
            
        # Determine target extension
        target_ext = self.SUPPORTED_FORMATS.get(output_format, ".flac")
        
        # Scan for files
        files_to_convert = []
        for root, _, files in os.walk(source_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                
                # Filter logic
                if input_filter != "All" and ext != input_filter.lower():
                    continue
                    
                # Skip non-audio files if "All" is selected (basic check)
                if input_filter == "All" and ext not in [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aiff", ".wma"]:
                    continue
                    
                files_to_convert.append(os.path.join(root, file))
                
        if not files_to_convert:
            return ["⚠️ No matching files found."]
            
        total_files = len(files_to_convert)
        logs.append(f"Found {total_files} files to convert.")
        
        success_count = 0
        
        for i, input_file in enumerate(files_to_convert):
            try:
                input_path = Path(input_file)
                
                # Determine Output Path
                if output_dir:
                    # Mirror structure
                    rel_path = input_path.relative_to(source_path)
                    dest_folder = Path(output_dir) / rel_path.parent
                    dest_folder.mkdir(parents=True, exist_ok=True)
                    output_file = dest_folder / (input_path.stem + target_ext)
                else:
                    # Same folder
                    output_file = input_path.with_suffix(target_ext)
                
                # Skip if input == output
                if input_path.resolve() == output_file.resolve():
                    logs.append(f"⏭️ Skipping {input_path.name} (Source matches Destination)")
                    continue
                    
                # Build FFmpeg Command
                cmd = [
                    "ffmpeg", "-y", # Overwrite
                    "-i", str(input_path),
                    "-map_metadata", "0", # Copy global metadata
                    "-id3v2_version", "3" # Compatibility for MP3
                ]
                
                # Format specific settings
                if output_format == "FLAC":
                    cmd.extend(["-compression_level", "5"])
                elif output_format == "MP3":
                    cmd.extend(["-b:a", "320k"]) # Max quality
                elif output_format == "M4A": # AAC
                    cmd.extend(["-c:a", "aac", "-b:a", "256k"])
                elif output_format == "ALAC":
                    cmd.extend(["-c:a", "alac"])
                
                cmd.append(str(output_file))
                
                # Run
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                success_count += 1
                logs.append(f"✅ Converted: {input_path.name} -> {output_file.name}")
                
            except subprocess.CalledProcessError as e:
                logs.append(f"❌ Failed: {input_path.name} (FFmpeg Error)")
                logger.error(f"FFmpeg failed for {input_path}: {e}")
            except Exception as e:
                logs.append(f"❌ Failed: {input_path.name} ({str(e)})")
                
            # Update Progress
            if progress_callback:
                p = (i + 1) / total_files
                progress_callback(p, f"Converting {i+1}/{total_files}: {input_path.name}")
                
        logs.append(f"--- Batch Complete. {success_count}/{total_files} successful. ---")
        return logs
