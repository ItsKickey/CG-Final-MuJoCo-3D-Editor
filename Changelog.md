# Version 1.0
Demo Version (Initial Release with Tkinter GUI)


# Version 1.1
ImGui Migration: Replaced Tkinter with ImGui to resolve thread conflicts and stabilize the rendering loop.

Core Refactoring: Rewrote `src/main_final.py` and `src/gui.py` files to fully adapt to the ImGui architecture.

Controls Update: Remapped camera rotation from Left Click to Right Click to prevent interference with UI interactions.

# Version 1.2
we now use the source code of obj2mjcf instead of using cmd line that might not avaliable on all the platform

## minor version 1.2.1 
fixed bug : when delete a duplicate mesh , it might delete all the texture of the same mesh
