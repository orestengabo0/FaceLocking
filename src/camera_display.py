# src/camera_display.py
"""
Camera display utilities for fullscreen/large resizable windows.

Provides helpers for creating large, resizable camera windows optimized
for Lenovo PC and similar high-resolution displays.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class CameraDisplay:
    """
    Manages fullscreen/large resizable camera windows.
    
    Features:
    - Fullscreen or large window creation
    - Resizable windows
    - Automatic scaling to screen resolution
    - Multiple window management
    """
    
    FULLSCREEN = "fullscreen"
    LARGE = "large"
    MEDIUM = "medium"
    
    def __init__(self, mode: str = LARGE):
        """
        Initialize camera display manager.
        
        Args:
            mode: "fullscreen", "large", or "medium"
        """
        self.mode = mode
        self.windows = {}
        
        # Get screen resolution (approximate for Lenovo/typical laptops)
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Window sizes based on mode
        if mode == self.FULLSCREEN:
            self.width = self.screen_width
            self.height = self.screen_height
        elif mode == self.LARGE:
            self.width = 1440
            self.height = 810
        else:  # medium
            self.width = 960
            self.height = 540
    
    def create_window(self, name: str, resizable: bool = True) -> None:
        """
        Create a large resizable window.
        
        Args:
            name: window name
            resizable: if True, window is resizable
        """
        flags = cv2.WINDOW_NORMAL if resizable else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(name, flags)
        
        if resizable:
            cv2.resizeWindow(name, self.width, self.height)
            
            # Move to center of screen (approximate)
            cv2.moveWindow(name, 0, 0)
        
        self.windows[name] = {
            "width": self.width,
            "height": self.height,
            "resizable": resizable
        }
    
    def show_frame(self, name: str, frame: np.ndarray) -> None:
        """
        Show frame in the specified window.
        
        Args:
            name: window name
            frame: BGR frame to display
        """
        if name not in self.windows:
            self.create_window(name)
        
        cv2.imshow(name, frame)
    
    def get_window_info(self, name: str) -> dict:
        """Get window information."""
        return self.windows.get(name, {})
    
    def close_window(self, name: str) -> None:
        """Close a specific window."""
        if name in self.windows:
            cv2.destroyWindow(name)
            del self.windows[name]
    
    def close_all(self) -> None:
        """Close all windows."""
        cv2.destroyAllWindows()
        self.windows = {}
    
    @staticmethod
    def scale_frame_to_window(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Scale frame to fit window while maintaining aspect ratio.
        
        Args:
            frame: input frame
            target_width: target window width
            target_height: target window height
        
        Returns:
            scaled frame
        """
        h, w = frame.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_width / w, target_height / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size if needed
        if new_w < target_width or new_h < target_height:
            top = (target_height - new_h) // 2
            left = (target_width - new_w) // 2
            bottom = target_height - new_h - top
            right = target_width - new_w - left
            
            scaled = cv2.copyMakeBorder(
                scaled,
                top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
        
        return scaled


# Global instance for convenience
_global_display: Optional[CameraDisplay] = None


def get_display(mode: str = CameraDisplay.LARGE) -> CameraDisplay:
    """Get or create global display manager."""
    global _global_display
    if _global_display is None:
        _global_display = CameraDisplay(mode=mode)
    return _global_display


def create_large_window(name: str, mode: str = CameraDisplay.LARGE) -> None:
    """
    Quick helper: create a large window.
    
    Args:
        name: window name
        mode: display mode (fullscreen, large, medium)
    """
    display = get_display(mode=mode)
    display.create_window(name)


def show_large_frame(name: str, frame: np.ndarray, mode: str = CameraDisplay.LARGE) -> None:
    """
    Quick helper: show frame in large window.
    
    Args:
        name: window name
        frame: BGR frame
        mode: display mode
    """
    display = get_display(mode=mode)
    display.show_frame(name, frame)


def close_large_windows() -> None:
    """Close all large windows."""
    global _global_display
    if _global_display:
        _global_display.close_all()
        _global_display = None